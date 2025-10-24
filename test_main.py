import unittest
import requests
import json
from multiprocessing import Process
import time
import uvicorn

# --- Test Data ---
SAMPLE_JOB_DATA = [
    ('job_1', '2025-09-26T10:00:00', '2025-09-26T10:00:05'),
    ('job_2', '2025-09-26T10:00:01', '2025-09-26T10:00:04'),
]

class TestApi(unittest.TestCase):

    BASE_URL = "http://127.0.0.1:8000"
    TOKEN = None

    @classmethod
    def setUpClass(cls):
        """Start the FastAPI server in a separate process."""

        def run_server():
            from main import app
            uvicorn.run(app, host="127.0.0.1", port=8000)

        cls.server_process = Process(target=run_server)
        cls.server_process.start()

        # Wait for the server to start
        for _ in range(10):
            try:
                response = requests.get(f"{cls.BASE_URL}/users/me/")
                if response.status_code == 401:
                    break
            except requests.ConnectionError:
                time.sleep(0.5)
        else:
            cls.tearDownClass()
            raise RuntimeError("Server did not start in time.")


    @classmethod
    def tearDownClass(cls):
        """Terminate the server process."""
        cls.server_process.terminate()
        cls.server_process.join()

    def test_01_login_for_access_token(self):
        """Tests the /token endpoint to get a JWT."""
        response = requests.post(
            f"{self.BASE_URL}/token",
            data={"username": "jules", "password": "test"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("access_token", response.json())
        self.assertIn("token_type", response.json())

        TestApi.TOKEN = response.json()["access_token"]


    def test_02_read_users_me_with_token(self):
        """Tests the /users/me endpoint with a valid token."""
        self.assertIsNotNone(TestApi.TOKEN, "Token not available for this test.")
        headers = {"Authorization": f"Bearer {TestApi.TOKEN}"}
        response = requests.get(f"{self.BASE_URL}/users/me/", headers=headers)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["username"], "jules")

    def test_03_read_users_me_no_token(self):
        """Tests that the /users/me endpoint returns 401 without a token."""
        response = requests.get(f"{self.BASE_URL}/users/me/")
        self.assertEqual(response.status_code, 401)

    def test_04_analyze_jobs_with_token(self):
        """Tests the /analyze endpoint with a valid token."""
        self.assertIsNotNone(TestApi.TOKEN, "Token not available for this test.")
        headers = {
            "Authorization": f"Bearer {TestApi.TOKEN}",
            "Content-Type": "application/json"
        }

        payload = {
            "job_data": SAMPLE_JOB_DATA,
            "num_workers": 2
        }

        response = requests.post(f"{self.BASE_URL}/analyze/", headers=headers, data=json.dumps(payload))

        self.assertEqual(response.status_code, 200)
        self.assertIn("total_job_count", response.json())
        self.assertEqual(response.json()["total_job_count"], 2)

    def test_05_analyze_jobs_no_token(self):
        """Tests that the /analyze endpoint returns 401 without a token."""
        headers = {"Content-Type": "application/json"}
        payload = {
            "job_data": SAMPLE_JOB_DATA,
            "num_workers": 2
        }

        response = requests.post(f"{self.BASE_URL}/analyze/", headers=headers, data=json.dumps(payload))
        self.assertEqual(response.status_code, 401)


if __name__ == '__main__':
    unittest.main()
