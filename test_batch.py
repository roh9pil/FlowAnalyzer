import unittest
import pandas as pd
from datetime import datetime, timedelta

from batch import predict_wait_time

class TestBatch(unittest.TestCase):
    def test_predict_wait_time(self):
        """
        Tests that the predict_wait_time function correctly calculates
        the average wait time and adds it to the DataFrame.
        """
        # 1. Create a sample DataFrame
        data = {
            'uid': ['job_1', 'job_2', 'job_3'],
            'created_on': [
                datetime(2025, 1, 1, 10, 0, 0),
                datetime(2025, 1, 1, 10, 0, 1),
                datetime(2025, 1, 1, 10, 0, 2),
            ],
            'updated_on': [
                datetime(2025, 1, 1, 10, 0, 5),
                datetime(2025, 1, 1, 10, 0, 4),
                datetime(2025, 1, 1, 10, 0, 8),
            ]
        }
        df = pd.DataFrame(data)

        # 2. Call the function to be tested
        result_df = predict_wait_time(df, num_workers=2)

        # 3. Assert the expected outcome
        self.assertIn('avg_wait_time', result_df.columns)

        # Calculation based on the logic in DelayedRequest.py:
        # job_1 (5s): worker1, starts 10:00:00, finishes 10:00:05. Wait: 0s.
        # job_2 (3s): worker2, starts 10:00:01, finishes 10:00:04. Wait: 0s.
        # job_3 (6s): worker2 is free at 10:00:04. Job submitted at 10:00:02.
        #             Starts at 10:00:04. Wait: 2s.
        # Total wait time: 0 + 0 + 2 = 2s
        # Average wait time: 2s / 3 jobs = 0.666...
        expected_avg_wait_time = 2.0 / 3.0

        self.assertAlmostEqual(result_df['avg_wait_time'].iloc[0], expected_avg_wait_time, places=6)
