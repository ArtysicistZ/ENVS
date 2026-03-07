import os
import sys
import unittest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from verl.trainer.ray_trainer import build_rollout_jobs, chunk_rollout_jobs, normalize_remote_server_urls


class TestRolloutScheduler(unittest.TestCase):
    def test_normalize_remote_server_urls_accepts_single_and_multiple(self):
        urls = normalize_remote_server_urls(
            "http://a:15001",
            ["http://b:15001", "http://a:15001", None],
        )
        self.assertEqual(urls, ["http://a:15001", "http://b:15001"])

    def test_build_rollout_jobs_preserves_grouped_task_order(self):
        tasks = [{"id": "t1"}, {"id": "t2"}, {"id": "t3"}]
        jobs = build_rollout_jobs(tasks, rollout_n=3)
        self.assertEqual([job["id"] for job in jobs], ["t1", "t1", "t1", "t2", "t2", "t2", "t3", "t3", "t3"])

    def test_chunk_rollout_jobs_collects_required_rollouts_when_envs_less_than_jobs(self):
        jobs = [{"idx": i} for i in range(10)]
        chunks = chunk_rollout_jobs(jobs, num_envs=4)
        self.assertEqual([len(chunk) for chunk in chunks], [4, 4, 2])
        flattened = [job["idx"] for chunk in chunks for job in chunk]
        self.assertEqual(flattened, list(range(10)))

    def test_chunk_rollout_jobs_handles_more_envs_than_rollouts(self):
        jobs = [{"idx": i} for i in range(3)]
        chunks = chunk_rollout_jobs(jobs, num_envs=8)
        self.assertEqual(len(chunks), 1)
        self.assertEqual([job["idx"] for job in chunks[0]], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
