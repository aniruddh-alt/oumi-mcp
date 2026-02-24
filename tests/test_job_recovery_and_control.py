import asyncio
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from oumi_mcp_server import job_service, server
from oumi_mcp_server.job_service import (
    JobRecord,
    JobRegistry,
    JobRuntime,
    cancel,
    get_runtime,
    make_job_id,
)


class JobRecoveryAndControlTests(unittest.IsolatedAsyncioTestCase):
    def test_registry_persists_and_rehydrates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_file = Path(tmp_dir) / "jobs.json"
            reg1 = JobRegistry(path=state_file)
            record = JobRecord(
                job_id="train_20260212_170925_abc123",
                command="train",
                config_path="/tmp/train.yaml",
                cloud="gcp",
                cluster_name="cluster-a",
                oumi_job_id="sky-job-123",
                model_name="meta-llama/Llama-3.1-8B-Instruct",
                status="running",
                submit_time="2026-02-12T17:09:25+00:00",
            )
            reg1.add(record)

            reg2 = JobRegistry(path=state_file)
            loaded = reg2.get(record.job_id)

        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.cloud, "gcp")
        self.assertEqual(loaded.cluster_name, "cluster-a")
        self.assertEqual(loaded.oumi_job_id, "sky-job-123")

    async def test_cancel_job_supports_direct_cloud_identity(self) -> None:
        with (
            patch("oumi_mcp_server.server._resolve_job_record", return_value=None),
            patch(
                "oumi_mcp_server.server.launcher.cancel", return_value=None
            ) as mock_cancel,
        ):
            response = await server.cancel_job(
                job_id="",
                oumi_job_id="sky-job-123",
                cloud="gcp",
                cluster_name="cluster-a",
            )

        self.assertTrue(response["success"])
        mock_cancel.assert_called_once_with("sky-job-123", "gcp", "cluster-a")

    async def test_cancel_job_returns_structured_error_on_launcher_failure(
        self,
    ) -> None:
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

    async def test_get_job_status_by_direct_identity_not_found_is_graceful(
        self,
    ) -> None:
        with (
            patch("oumi_mcp_server.server._resolve_job_record", return_value=None),
            patch(
                "oumi_mcp_server.server._fetch_cloud_status_direct", return_value=None
            ),
        ):
            response = await server.get_job_status(
                job_id="",
                oumi_job_id="sky-job-123",
                cloud="gcp",
                cluster_name="cluster-a",
            )

        self.assertFalse(response["success"])
        self.assertEqual(response["status"], "not_found")

    async def test_get_job_logs_by_direct_identity_returns_helpful_message(
        self,
    ) -> None:
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

    async def test_run_oumi_job_blocks_malformed_yaml_at_execution_boundary(
        self,
    ) -> None:
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

    async def test_dry_run_cloud_warns_about_missing_env_vars(self) -> None:
        """Dry-run for cloud job should warn when local env vars won't be forwarded."""
        import os

        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = Path(tmp_dir) / "train.yaml"
            cfg.write_text("model: {model_name: test/model}\n", encoding="utf-8")
            with patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}, clear=False):
                response = await server.run_oumi_job(
                    config_path=str(cfg),
                    command="train",
                    cloud="gcp",
                    dry_run=True,
                    confirm=False,
                )
        self.assertTrue(response["success"])
        self.assertTrue(response["dry_run"])
        self.assertIn("WANDB_API_KEY", response["message"])

    async def test_cancel_pending_cloud_launch_marks_intent(self) -> None:
        record = JobRecord(
            job_id="job-1",
            command="train",
            config_path="/tmp/train.yaml",
            cloud="gcp",
            cluster_name="",
            oumi_job_id="",
            model_name="",
            status="running",
            submit_time="2026-02-12T17:09:25+00:00",
        )
        rt = JobRuntime()
        with patch("oumi_mcp_server.job_service.get_registry") as mock_registry:
            mock_registry.return_value.update = lambda *a, **kw: None
            response = await job_service.cancel(record, rt)
        self.assertTrue(response["success"])
        self.assertTrue(rt.cancel_requested)

    async def test_cloud_launch_uses_staged_config_and_working_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_path = Path(tmp_dir) / "train.yaml"
            cfg_path.write_text("model: {model_name: test/model}\n", encoding="utf-8")
            record = JobRecord(
                job_id="train_20260220_000001_abc123",
                command="train",
                config_path=str(cfg_path),
                cloud="gcp",
                cluster_name="",
                oumi_job_id="",
                model_name="",
                status="running",
                submit_time="2026-02-20T00:00:01+00:00",
            )
            rt = JobRuntime()
            rt.run_dir = Path(tmp_dir) / "run"

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
                    mock_registry.return_value.update = lambda *a, **kw: None
                    await job_service._launch_cloud(record, rt, accelerators="A100:1")  # type: ignore[attr-defined]

            job_cfg = captured_job_cfg["cfg"]
            self.assertEqual(Path(job_cfg.working_dir), rt.run_dir)
            self.assertIn("-c config.yaml", job_cfg.run)
            self.assertIn("command -v oumi", job_cfg.run)
            self.assertIn("export PATH=", job_cfg.run)
            self.assertIn("uv pip install --system", job_cfg.setup or "")
            self.assertTrue((rt.run_dir / "config.yaml").exists())

    async def test_cloud_launch_reconciles_pending_cancel_after_id_available(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_path = Path(tmp_dir) / "train.yaml"
            cfg_path.write_text("model: {model_name: test/model}\n", encoding="utf-8")
            record = JobRecord(
                job_id="train_20260220_000002_def456",
                command="train",
                config_path=str(cfg_path),
                cloud="gcp",
                cluster_name="",
                oumi_job_id="",
                model_name="",
                status="running",
                submit_time="2026-02-20T00:00:02+00:00",
            )
            rt = JobRuntime()
            rt.cancel_requested = True
            rt.run_dir = Path(tmp_dir) / "run"

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
            # Set up a mock registry that applies updates to the real record
            # so _launch_cloud can refresh it after updating.
            def _mock_update(job_id, **fields):  # noqa: ANN001, ANN003
                for k, v in fields.items():
                    setattr(record, k, v)

            mock_reg = SimpleNamespace(
                update=_mock_update,
                get=lambda jid: record,
            )
            with (
                patch(
                    "oumi_mcp_server.job_service.launcher.up",
                    return_value=(SimpleNamespace(), status),
                ),
                patch(
                    "oumi_mcp_server.job_service.launcher.cancel",
                    return_value=cancelled,
                ) as mock_cancel,
                patch(
                    "oumi_mcp_server.job_service.get_registry",
                    return_value=mock_reg,
                ),
            ):
                await job_service._launch_cloud(record, rt)  # type: ignore[attr-defined]

            mock_cancel.assert_called_once_with("cloud-456", "gcp", "cluster-b")
            self.assertTrue(rt.cancel_requested)

    def test_read_log_tail_returns_last_n_lines_for_large_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "big.log"
            lines = [f"line-{idx}" for idx in range(1, 20001)]
            log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            tail, count = server._read_log_tail(log_path, 5)  # type: ignore[attr-defined]
        self.assertEqual(count, 5)
        self.assertEqual(tail.splitlines(), lines[-5:])

    def test_default_cloud_setup_script_has_no_version_pins(self) -> None:
        """Default setup script must not pin oumi to a version range."""
        from oumi_mcp_server.job_service import _DEFAULT_CLOUD_SETUP_SCRIPT
        self.assertNotIn(">=", _DEFAULT_CLOUD_SETUP_SCRIPT)
        self.assertNotIn("<=", _DEFAULT_CLOUD_SETUP_SCRIPT)
        self.assertNotIn("<0.", _DEFAULT_CLOUD_SETUP_SCRIPT)
        self.assertIn("oumi[gpu]", _DEFAULT_CLOUD_SETUP_SCRIPT)
        self.assertIn("command -v oumi", _DEFAULT_CLOUD_SETUP_SCRIPT)

    def test_cluster_lifecycle_response_is_importable(self) -> None:
        from oumi_mcp_server.models import ClusterLifecycleResponse
        r: ClusterLifecycleResponse = {"success": True, "message": "ok"}
        self.assertTrue(r["success"])

    async def test_dry_run_cloud_shows_jobconfig_yaml_preview(self) -> None:
        """Cloud dry-run for training config should show full JobConfig YAML."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = Path(tmp_dir) / "train.yaml"
            cfg.write_text("model: {model_name: test/model}\n", encoding="utf-8")
            response = await server.run_oumi_job(
                config_path=str(cfg),
                command="train",
                cloud="gcp",
                accelerators="A100:4",
                dry_run=True,
            )
        self.assertTrue(response["success"])
        msg = response["message"]
        self.assertIn("resources:", msg)
        self.assertIn("gcp", msg)
        self.assertIn("A100:4", msg)
        self.assertIn("setup:", msg)
        self.assertIn("run:", msg)
        self.assertIn("--- Generated JobConfig", msg)


    async def test_stop_cluster_calls_launcher_stop(self) -> None:
        with patch("oumi_mcp_server.server.launcher.stop") as mock_stop:
            response = await server.stop_cluster(cloud="gcp", cluster_name="sky-xxxx")
        self.assertTrue(response["success"])
        mock_stop.assert_called_once_with("gcp", "sky-xxxx")

    async def test_stop_cluster_returns_error_on_failure(self) -> None:
        with patch(
            "oumi_mcp_server.server.launcher.stop",
            side_effect=RuntimeError("network error"),
        ):
            response = await server.stop_cluster(cloud="gcp", cluster_name="sky-xxxx")
        self.assertFalse(response["success"])
        self.assertIn("Failed to stop cluster", response.get("error", ""))

    async def test_stop_cluster_rejects_empty_args(self) -> None:
        response = await server.stop_cluster(cloud="", cluster_name="sky-xxxx")
        self.assertFalse(response["success"])
        self.assertIn("required", response.get("error", ""))

    async def test_down_cluster_without_confirm_returns_dryrun_message(self) -> None:
        with patch("oumi_mcp_server.server.launcher.down") as mock_down:
            response = await server.down_cluster(cloud="gcp", cluster_name="sky-xxxx")
        mock_down.assert_not_called()
        self.assertTrue(response["success"])
        self.assertIn("IRREVERSIBLE", response.get("message", ""))

    async def test_down_cluster_with_confirm_calls_launcher_down(self) -> None:
        with patch("oumi_mcp_server.server.launcher.down") as mock_down:
            response = await server.down_cluster(
                cloud="gcp",
                cluster_name="sky-xxxx",
                confirm=True,
                user_confirmation="DOWN",
            )
        self.assertTrue(response["success"])
        mock_down.assert_called_once_with("gcp", "sky-xxxx")

    async def test_down_cluster_wrong_confirmation_phrase_is_blocked(self) -> None:
        with patch("oumi_mcp_server.server.launcher.down") as mock_down:
            response = await server.down_cluster(
                cloud="gcp",
                cluster_name="sky-xxxx",
                confirm=True,
                user_confirmation="EXECUTE",
            )
        mock_down.assert_not_called()
        self.assertFalse(response["success"])

    async def test_down_cluster_returns_error_on_failure(self) -> None:
        with patch(
            "oumi_mcp_server.server.launcher.down",
            side_effect=RuntimeError("cloud error"),
        ):
            response = await server.down_cluster(
                cloud="gcp",
                cluster_name="sky-xxxx",
                confirm=True,
                user_confirmation="DOWN",
            )
        self.assertFalse(response["success"])
        self.assertIn("Failed to delete cluster", response.get("error", ""))


    def test_get_started_mentions_all_new_tools(self) -> None:
        result = server.get_started()
        self.assertIn("stop_cluster", result)
        self.assertIn("down_cluster", result)
        self.assertIn("Cloud Job Workflow", result)
        self.assertIn("Cluster Lifecycle", result)
        self.assertIn("suggested_configs", result)


    async def test_cancel_pending_cloud_job_cancels_runner_task(self) -> None:
        """cancel() should call runner_task.cancel() for pre-launch cloud jobs."""
        record = JobRecord(
            job_id="test-cancel-task",
            command="train",
            config_path="/tmp/fake.yaml",
            cloud="gcp",
            cluster_name="",
            oumi_job_id="",
            model_name="",
            status="running",
            submit_time="2026-02-12T17:09:25+00:00",
        )
        rt = JobRuntime()
        mock_task = asyncio.Future()
        rt.runner_task = mock_task  # type: ignore[assignment]
        with patch("oumi_mcp_server.job_service.get_registry") as mock_registry:
            mock_registry.return_value.update = lambda *a, **kw: None
            result = await cancel(record, rt)
        self.assertTrue(result["success"])
        self.assertTrue(mock_task.cancelled())

    def test_make_job_id_sanitizes_path_traversal(self) -> None:
        """make_job_id should strip path traversal characters from job_name."""
        self.assertNotIn("/", make_job_id("train", job_name="../../etc/evil"))
        self.assertNotIn("\\", make_job_id("train", job_name="..\\..\\evil"))
        self.assertNotIn("..", make_job_id("train", job_name="../up"))

    def test_make_job_id_rejects_empty_after_sanitization(self) -> None:
        """make_job_id should raise ValueError if job_name is only unsafe chars."""
        with self.assertRaises(ValueError):
            make_job_id("train", job_name="../../..")


if __name__ == "__main__":
    unittest.main()
