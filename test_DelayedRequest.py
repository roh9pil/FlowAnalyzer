# Tests for DelayedRequest.py
import unittest
from datetime import datetime, timedelta
import numpy as np

from DelayedRequest import (
    _prepare_jobs,
    _run_simulation,
    _calculate_metrics,
    analyze_job_queue,
    Job
)

# A small constant to handle floating-point comparisons.
EPSILON = 1e-9

class TestPrepareJobs(unittest.TestCase):
    def setUp(self):
        self.timestamp_format = "%Y-%m-%dT%H:%M:%S"

    def test_prepare_jobs_valid(self):
        """Tests that jobs are prepared and sorted correctly."""
        job_data = [
            ('job_2', '2025-01-01T10:00:01', '2025-01-01T10:00:04'),
            ('job_1', '2025-01-01T10:00:00', '2025-01-01T10:00:05'),
        ]

        jobs = _prepare_jobs(job_data, self.timestamp_format)

        self.assertEqual(len(jobs), 2)
        self.assertEqual(jobs[0].uid, 'job_1') # Check sorting
        self.assertEqual(jobs[1].uid, 'job_2')
        self.assertIsInstance(jobs[0], Job)
        self.assertEqual(jobs[0].duration, 5)

    def test_prepare_jobs_invalid_time_range(self):
        """Tests that jobs with end_time < start_time are filtered out."""
        job_data = [
            ('valid_job', '2025-01-01T10:00:00', '2025-01-01T10:00:05'),
            ('invalid_job', '2025-01-01T10:00:10', '2025-01-01T10:00:05'), # Invalid
        ]

        jobs = _prepare_jobs(job_data, self.timestamp_format)

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].uid, 'valid_job')

    def test_prepare_jobs_invalid_format(self):
        """Tests that a ValueError is raised for incorrect timestamp formats."""
        job_data = [
            ('job_1', '2025-01-01 10:00:00', '2025-01-01 10:00:05'), # Wrong format
        ]

        with self.assertRaises(ValueError):
            _prepare_jobs(job_data, self.timestamp_format)


class TestRunSimulation(unittest.TestCase):
    def test_run_simulation_single_worker(self):
        """Tests the simulation with a single worker and staggered jobs."""
        # job1: starts 10:00:00, duration 5s -> finishes 10:00:05
        # job2: starts 10:00:01, must wait for job1 -> starts 10:00:05, wait 4s
        jobs = [
            Job('job_1', 1736090400.0, 1736090405.0, 5), # 2025-01-01T10:00:00
            Job('job_2', 1736090401.0, 1736090404.0, 3), # 2025-01-01T10:00:01
        ]

        wait_times, delayed_count = _run_simulation(jobs, num_workers=1)

        self.assertAlmostEqual(wait_times[0], 0.0, delta=EPSILON)
        self.assertAlmostEqual(wait_times[1], 4.0, delta=EPSILON)
        self.assertEqual(delayed_count, 1)

    def test_run_simulation_multiple_workers(self):
        """Tests that multiple workers handle jobs concurrently."""
        # job1 (5s): worker1, starts 10:00:00, finishes 10:00:05
        # job2 (3s): worker2, starts 10:00:01, finishes 10:00:04
        # job3 (6s): worker2 free at 10:00:04, starts 10:00:04, wait 2s
        jobs = [
            Job('job_1', 1736090400.0, 1736090405.0, 5), # 10:00:00
            Job('job_2', 1736090401.0, 1736090404.0, 3), # 10:00:01
            Job('job_3', 1736090402.0, 1736090408.0, 6), # 10:00:02
        ]

        wait_times, delayed_count = _run_simulation(jobs, num_workers=2)

        self.assertAlmostEqual(wait_times[0], 0.0, delta=EPSILON) # job1 -> worker1
        self.assertAlmostEqual(wait_times[1], 0.0, delta=EPSILON) # job2 -> worker2
        self.assertAlmostEqual(wait_times[2], 2.0, delta=EPSILON) # job3 waits
        self.assertEqual(delayed_count, 1)

    def test_run_simulation_no_delay(self):
        """Tests a scenario where no jobs should be delayed."""
        # Jobs are spaced out enough that a worker is always free.
        jobs = [
            Job('job_1', 1736090400.0, 1736090402.0, 2), # 10:00:00
            Job('job_2', 1736090405.0, 1736090407.0, 2), # 10:00:05
            Job('job_3', 1736090410.0, 1736090412.0, 2), # 10:00:10
        ]

        wait_times, delayed_count = _run_simulation(jobs, num_workers=1)

        self.assertAlmostEqual(sum(wait_times), 0.0, delta=EPSILON)
        self.assertEqual(delayed_count, 0)


class TestCalculateMetrics(unittest.TestCase):
    def test_calculate_metrics_valid(self):
        """Tests the calculation of performance metrics."""
        wait_times = [0.0, 4.0, 2.0]
        total_jobs = 10
        delayed_jobs = 2 # Note: this is different from len(wait_times)

        metrics = _calculate_metrics(wait_times, total_jobs, delayed_jobs)

        self.assertAlmostEqual(metrics['total_wait_time'], 6.0, delta=EPSILON)
        self.assertAlmostEqual(metrics['average_wait_time'], 2.0, delta=EPSILON)
        self.assertAlmostEqual(metrics['variance'], np.var(wait_times), delta=EPSILON)
        self.assertAlmostEqual(metrics['std_deviation'], np.std(wait_times), delta=EPSILON)
        self.assertEqual(metrics['delayed_job_count'], delayed_jobs)
        self.assertAlmostEqual(metrics['delayed_jobs_ratio'], 0.2, delta=EPSILON)
        self.assertEqual(metrics['total_job_count'], total_jobs)

    def test_calculate_metrics_empty(self):
        """Tests metric calculation with no wait times."""
        metrics = _calculate_metrics([], 0, 0)

        self.assertEqual(metrics['total_wait_time'], 0.0)
        self.assertEqual(metrics['average_wait_time'], 0.0)
        self.assertEqual(metrics['variance'], 0.0)
        self.assertEqual(metrics['delayed_job_count'], 0)
        self.assertEqual(metrics['delayed_jobs_ratio'], 0.0)
        self.assertEqual(metrics['total_job_count'], 0)


class TestAnalyzeJobQueue(unittest.TestCase):
    def setUp(self):
        self.timestamp_format = "%Y-%m-%dT%H:%M:%S"
        self.sample_job_data = [
            ('job_1', '2025-01-01T10:00:00', '2025-01-01T10:00:05'), # 5s
            ('job_2', '2025-01-01T10:00:01', '2025-01-01T10:00:04'), # 3s
            ('job_3', '2025-01-01T10:00:02', '2025-01-01T10:00:08'), # 6s
        ]

    def test_analyze_job_queue_end_to_end(self):
        """Tests the full analysis pipeline with a valid dataset."""
        results = analyze_job_queue(
            self.sample_job_data,
            num_workers=2,
            timestamp_format=self.timestamp_format
        )

        self.assertEqual(results['total_job_count'], 3)
        self.assertEqual(results['delayed_job_count'], 1)
        self.assertAlmostEqual(results['average_wait_time'], 2.0 / 3.0, delta=EPSILON)

    def test_analyze_job_queue_empty_jobs(self):
        """Tests the analyzer with an empty list of jobs."""
        results = analyze_job_queue([], 2, self.timestamp_format)
        self.assertEqual(results['total_job_count'], 0)
        self.assertEqual(results['total_wait_time'], 0.0)

    def test_analyze_job_queue_zero_workers(self):
        """Tests the analyzer with zero workers."""
        results = analyze_job_queue(self.sample_job_data, 0, self.timestamp_format)
        self.assertEqual(results['total_job_count'], 0)
        self.assertEqual(results['total_wait_time'], 0.0)

    def test_analyze_job_queue_invalid_timestamp(self):
        """Tests that an invalid timestamp format returns an empty dictionary."""
        invalid_data = [('job_1', 'invalid-date', '2025-01-01T10:00:05')]
        results = analyze_job_queue(invalid_data, 2, self.timestamp_format)
        self.assertEqual(results, {})

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)