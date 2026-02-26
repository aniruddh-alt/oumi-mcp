import json
import tempfile
import unittest
from pathlib import Path

from oumi_mcp_server.job_service import JobRecord, JobRegistry, JobRuntime


class TestJobRecord(unittest.TestCase):
    def test_all_fields_are_strings(self):
        r = JobRecord(
            job_id="train_001",
            command="train",
            config_path="/tmp/train.yaml",
            cloud="gcp",
            cluster_name="cluster-a",
            oumi_job_id="sky-123",
            model_name="meta-llama/Llama-3.1-8B",
            submit_time="2026-02-24T12:00:00Z",
        )
        for field_name in ["job_id", "command", "config_path", "cloud",
                           "cluster_name", "oumi_job_id", "model_name",
                           "submit_time"]:
            self.assertIsInstance(getattr(r, field_name), str)


class TestJobRuntime(unittest.TestCase):
    def test_defaults_are_none(self):
        rt = JobRuntime()
        self.assertIsNone(rt.process)
        self.assertIsNone(rt.cluster_obj)
        self.assertIsNone(rt.runner_task)
        self.assertIsNone(rt.oumi_status)


class TestJobRegistry(unittest.TestCase):
    def test_add_and_get(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.json"
            reg = JobRegistry(path)
            r = JobRecord(
                job_id="j1", command="train", config_path="/tmp/t.yaml",
                cloud="local", cluster_name="", oumi_job_id="123",
                model_name="test",
                submit_time="2026-02-24T00:00:00Z",
            )
            reg.add(r)
            self.assertEqual(reg.get("j1").job_id, "j1")

    def test_persists_to_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.json"
            reg = JobRegistry(path)
            r = JobRecord(
                job_id="j1", command="train", config_path="/tmp/t.yaml",
                cloud="gcp", cluster_name="c1", oumi_job_id="sky-1",
                model_name="test",
                submit_time="2026-02-24T00:00:00Z",
            )
            reg.add(r)
            # Load a new registry from the same file
            reg2 = JobRegistry(path)
            loaded = reg2.get("j1")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.cloud, "gcp")
            self.assertEqual(loaded.oumi_job_id, "sky-1")

    def test_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.json"
            reg = JobRegistry(path)
            r = JobRecord(
                job_id="j1", command="train", config_path="/tmp/t.yaml",
                cloud="local", cluster_name="", oumi_job_id="",
                model_name="test",
                submit_time="2026-02-24T00:00:00Z",
            )
            reg.add(r)
            reg.update("j1", oumi_job_id="456")
            self.assertEqual(reg.get("j1").oumi_job_id, "456")

    def test_remove(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.json"
            reg = JobRegistry(path)
            r = JobRecord(
                job_id="j1", command="train", config_path="/tmp/t.yaml",
                cloud="local", cluster_name="", oumi_job_id="",
                model_name="test",
                submit_time="2026-02-24T00:00:00Z",
            )
            reg.add(r)
            reg.remove("j1")
            self.assertIsNone(reg.get("j1"))
            # Verify persisted
            reg2 = JobRegistry(path)
            self.assertIsNone(reg2.get("j1"))

    def test_find_by_cloud_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.json"
            reg = JobRegistry(path)
            r = JobRecord(
                job_id="j1", command="train", config_path="/tmp/t.yaml",
                cloud="gcp", cluster_name="c1", oumi_job_id="sky-99",
                model_name="test",
                submit_time="2026-02-24T00:00:00Z",
            )
            reg.add(r)
            found = reg.find_by_cloud_identity("gcp", "sky-99")
            self.assertEqual(found.job_id, "j1")
            self.assertIsNone(reg.find_by_cloud_identity("aws", "sky-99"))

    def test_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.json"
            reg = JobRegistry(path)
            for i in range(3):
                reg.add(JobRecord(
                    job_id=f"j{i}", command="train", config_path="/tmp/t.yaml",
                    cloud="local", cluster_name="", oumi_job_id=str(i),
                    model_name="test",
                    submit_time="2026-02-24T00:00:00Z",
                ))
            self.assertEqual(len(reg.all()), 3)

    def test_load_corrupt_file_starts_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.json"
            path.write_text("not valid json{{{", encoding="utf-8")
            reg = JobRegistry(path)
            self.assertEqual(len(reg.all()), 0)

    def test_load_missing_file_starts_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.json"
            reg = JobRegistry(path)
            self.assertEqual(len(reg.all()), 0)


if __name__ == "__main__":
    unittest.main()
