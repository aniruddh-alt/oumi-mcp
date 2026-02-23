import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from oumi_mcp_server import job_service, server
from oumi_mcp_server.job_service import JobRecord, JobRegistry


class JobRecoveryAndControlTests(unittest.IsolatedAsyncioTestCase):
    async def test_registry_rehydrates_from_persisted_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_dir = Path(tmp_dir)
            with patch.object(job_service, "JOB_STATE_DIR", state_dir):
                reg1 = JobRegistry()
                record = JobRecord(
                    job_id="train_20260212_170925_abc123",
                    command="train",
                    config_path="/tmp/train.yaml",
                    cloud="gcp",
                    cluster_name="cluster-a",
                    oumi_job_id="sky-job-123",
                    model_name="meta-llama/Llama-3.1-8B-Instruct",
                    output_dir="/tmp/output",
                )
                await reg1.register(record)

                reg2 = JobRegistry()
                loaded = await reg2.get(record.job_id)

        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.cloud, "gcp")
        self.assertEqual(loaded.cluster_name, "cluster-a")
        self.assertEqual(loaded.oumi_job_id, "sky-job-123")

    async def test_cancel_job_supports_direct_cloud_identity(self) -> None:
        with (
            patch("oumi_mcp_server.server._resolve_job_record", return_value=None),
            patch("oumi_mcp_server.server.launcher.cancel", return_value=None) as mock_cancel,
        ):
            response = await server.cancel_job(
                job_id="",
                oumi_job_id="sky-job-123",
                cloud="gcp",
                cluster_name="cluster-a",
            )

        self.assertTrue(response["success"])
        mock_cancel.assert_called_once_with("sky-job-123", "gcp", "cluster-a")

    async def test_cancel_job_returns_structured_error_on_launcher_failure(self) -> None:
        with (
            patch("oumi_mcp_server.server._resolve_job_record", return_value=None),
            patch(
                "oumi_mcp_server.server.launcher.cancel",
                side_effect=RuntimeError("cancel failed"),
            ),
        ):
            response = await server.cancel_job(
                job_id="",
                oumi_job_id="sky-job-123",
                cloud="gcp",
                cluster_name="cluster-a",
            )

        self.assertFalse(response["success"])
        self.assertIn("Failed to cancel cloud job", response["error"])

    async def test_get_job_status_by_direct_identity_not_found_is_graceful(self) -> None:
        with (
            patch("oumi_mcp_server.server._resolve_job_record", return_value=None),
            patch("oumi_mcp_server.server._fetch_cloud_status_direct", return_value=None),
        ):
            response = await server.get_job_status(
                job_id="",
                oumi_job_id="sky-job-123",
                cloud="gcp",
                cluster_name="cluster-a",
            )

        self.assertFalse(response["success"])
        self.assertEqual(response["status"], "not_found")

    async def test_get_job_logs_by_direct_identity_returns_helpful_message(self) -> None:
        with patch("oumi_mcp_server.server._resolve_job_record", return_value=None):
            response = await server.get_job_logs(
                job_id="",
                oumi_job_id="sky-job-123",
                cloud="gcp",
                cluster_name="cluster-a",
                lines=50,
            )

        self.assertFalse(response["success"])
        self.assertIn("sky logs cluster-a", response["error"])

    def test_logging_configuration_downgrades_noisy_mcp_loggers(self) -> None:
        server._configure_logging()
        self.assertEqual(
            logging.getLogger("mcp.server.lowlevel.server").level,
            logging.WARNING,
        )

    async def test_run_oumi_job_blocks_malformed_yaml_at_execution_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_cfg = Path(tmp_dir) / "bad.yaml"
            bad_cfg.write_text("model: [", encoding="utf-8")
            response = await server.run_oumi_job(
                config_path=str(bad_cfg),
                command="train",
                dry_run=False,
                confirm=True,
                user_confirmation="EXECUTE",
            )
        self.assertFalse(response["success"])
        self.assertIn("Invalid YAML config", response["error"])

    async def test_cancel_pending_cloud_launch_marks_intent(self) -> None:
        record = JobRecord(
            job_id="job-1",
            command="train",
            config_path="/tmp/train.yaml",
            cloud="gcp",
        )
        with patch("oumi_mcp_server.job_service.get_registry") as mock_registry:
            mock_registry.return_value.persist = AsyncMock(return_value=None)
            response = await job_service.cancel(record)
        self.assertTrue(response["success"])
        self.assertTrue(record.cancel_requested)
        self.assertEqual(record.launch_state, "cancel_requested")

    async def test_cloud_launch_uses_staged_config_and_working_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_path = Path(tmp_dir) / "train.yaml"
            cfg_path.write_text("model: {model_name: test/model}\n", encoding="utf-8")
            record = JobRecord(
                job_id="train_20260220_000001_abc123",
                command="train",
                config_path=str(cfg_path),
                cloud="gcp",
            )

            captured_job_cfg = {}

            def _fake_up(job_cfg, cluster_name):  # noqa: ANN001
                captured_job_cfg["cfg"] = job_cfg
                captured_job_cfg["cluster"] = cluster_name
                status = SimpleNamespace(
                    id="cloud-123",
                    cluster="cluster-a",
                    done=False,
                    status="RUNNING",
                    state=SimpleNamespace(name="RUNNING"),
                    metadata={},
                )
                return (SimpleNamespace(), status)

            with patch("oumi_mcp_server.job_service.launcher.up", side_effect=_fake_up):
                with patch("oumi_mcp_server.job_service.get_registry") as mock_registry:
                    mock_registry.return_value.persist = AsyncMock(return_value=None)
                    await job_service._launch_cloud(record, accelerators="A100:1")  # type: ignore[attr-defined]

            job_cfg = captured_job_cfg["cfg"]
            self.assertEqual(Path(job_cfg.working_dir), record.run_dir)
            # Config path must be the relative filename — SkyPilot sets cwd to working_dir
            self.assertIn("-c config.yaml", job_cfg.run)
            # PATH extension and binary check must be present in run script
            self.assertIn("command -v oumi", job_cfg.run)
            self.assertIn("export PATH=", job_cfg.run)
            self.assertIn("uv pip install --system", job_cfg.setup or "")
            self.assertTrue((record.run_dir / "config.yaml").exists())

    async def test_cloud_launch_reconciles_pending_cancel_after_id_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_path = Path(tmp_dir) / "train.yaml"
            cfg_path.write_text("model: {model_name: test/model}\n", encoding="utf-8")
            record = JobRecord(
                job_id="train_20260220_000002_def456",
                command="train",
                config_path=str(cfg_path),
                cloud="gcp",
                cancel_requested=True,
            )

            status = SimpleNamespace(
                id="cloud-456",
                cluster="cluster-b",
                done=False,
                status="RUNNING",
                state=SimpleNamespace(name="RUNNING"),
                metadata={},
            )
            cancelled = SimpleNamespace(
                id="cloud-456",
                cluster="cluster-b",
                done=True,
                status="CANCELLED",
                state=SimpleNamespace(name="CANCELLED"),
                metadata={},
            )
            with (
                patch("oumi_mcp_server.job_service.launcher.up", return_value=(SimpleNamespace(), status)),
                patch("oumi_mcp_server.job_service.launcher.cancel", return_value=cancelled) as mock_cancel,
                patch("oumi_mcp_server.job_service.get_registry") as mock_registry,
            ):
                mock_registry.return_value.persist = AsyncMock(return_value=None)
                await job_service._launch_cloud(record)  # type: ignore[attr-defined]

            mock_cancel.assert_called_once_with("cloud-456", "gcp", "cluster-b")
            self.assertEqual(record.launch_state, "cancel_requested")

    def test_read_log_tail_returns_last_n_lines_for_large_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "big.log"
            lines = [f"line-{idx}" for idx in range(1, 20001)]
            log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            tail, count = server._read_log_tail(log_path, 5)  # type: ignore[attr-defined]
        self.assertEqual(count, 5)
        self.assertEqual(tail.splitlines(), lines[-5:])


if __name__ == "__main__":
    unittest.main()
