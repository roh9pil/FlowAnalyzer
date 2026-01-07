import heapq
import numpy as np
from collections import namedtuple
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import os
import csv

# A simple data structure for a job.
# Timestamps are stored as numeric Unix timestamps (seconds since epoch).
Job = namedtuple('Job', ['uid', 'start_time', 'end_time', 'duration'])

# A small constant to handle floating-point comparisons.
EPSILON = 1e-9


def _prepare_jobs(job_data: List[Tuple[str, str, str]], timestamp_format: str) -> List[Job]:
    """
    Parses raw job data into a sorted list of Job objects.
    Filters out jobs with invalid start/end times.
    """
    jobs = []
    try:
        for uid, start_str, end_str in job_data:
            start_dt = datetime.strptime(start_str, timestamp_format)
            end_dt = datetime.strptime(end_str, timestamp_format)
            start_ts = start_dt.timestamp()
            end_ts = end_dt.timestamp()

            if start_ts > end_ts:
                print(f"Warning: Job '{uid}' has start_time > end_time. Skipping.")
                continue
            
            duration = end_ts - start_ts
            jobs.append(Job(uid, start_ts, end_ts, duration))
    except ValueError as e:
        print(f"Error parsing timestamp. Please check the timestamp_format.")
        print(f"Details: {e}")
        raise  # Re-raise to be handled by the main analysis function

    # Sort jobs by their start time for sequential processing
    jobs.sort(key=lambda x: x.start_time)
    return jobs


def _run_simulation(
    jobs: List[Job],
    initial_num_workers: int,
    parallelism_map: Optional[Dict[str, int]] = None
) -> Tuple[List[float], int]:
    """
    Runs the core job queue simulation logic with dynamic worker counts.
    """
    first_job_start_time = jobs[0].start_time
    worker_finish_times = [first_job_start_time] * initial_num_workers
    heapq.heapify(worker_finish_times)
    
    wait_times = []
    delayed_job_count = 0

    current_date_str = datetime.fromtimestamp(first_job_start_time).strftime('%Y-%m-%d')

    for job in jobs:
        job_date_str = datetime.fromtimestamp(job.start_time).strftime('%Y-%m-%d')

        # Adjust worker pool if the date has changed
        if parallelism_map and job_date_str != current_date_str:
            current_date_str = job_date_str
            new_worker_count = parallelism_map.get(job_date_str, len(worker_finish_times))

            # Sync worker times before resizing the pool
            current_time = job.start_time
            synced_finish_times = [max(t, current_time) for t in worker_finish_times]

            # Resize the worker pool
            if new_worker_count > len(synced_finish_times):
                # Add new workers, available immediately
                synced_finish_times.extend([current_time] * (new_worker_count - len(synced_finish_times)))
            elif new_worker_count < len(synced_finish_times):
                # Remove workers that will finish earliest
                synced_finish_times.sort(reverse=True) # Sort to remove earliest finishers
                synced_finish_times = synced_finish_times[:new_worker_count]

            # Re-heapify the list to maintain the heap invariant
            worker_finish_times = synced_finish_times
            heapq.heapify(worker_finish_times)

        earliest_worker_free_time = heapq.heappop(worker_finish_times)
        actual_start_time = max(job.start_time, earliest_worker_free_time)
        wait_time = actual_start_time - job.start_time
        wait_times.append(wait_time)

        if wait_time > EPSILON:
            delayed_job_count += 1

        finish_time = actual_start_time + job.duration
        heapq.heappush(worker_finish_times, finish_time)

    return wait_times, delayed_job_count


def _calculate_metrics(wait_times: List[float], total_jobs: int, delayed_jobs: int) -> Dict[str, Any]:
    """Calculates performance metrics from simulation results."""
    if not wait_times:
        return {
            "total_wait_time": 0.0,
            "average_wait_time": 0.0,
            "variance": 0.0,
            "std_deviation": 0.0,
            "wait_times_list": [],
            "delayed_job_count": 0,
            "delayed_jobs_ratio": 0.0,
            "total_job_count": 0
        }

    total_wait_time = np.sum(wait_times)
    average_wait_time = np.mean(wait_times)
    variance = np.var(wait_times)
    std_deviation = np.std(wait_times)

    delayed_jobs_ratio = delayed_jobs / total_jobs if total_jobs > 0 else 0.0

    return {
        "total_wait_time": total_wait_time,
        "average_wait_time": average_wait_time,
        "variance": variance,
        "std_deviation": std_deviation,
        "wait_times_list": wait_times,
        "delayed_job_count": delayed_jobs,
        "delayed_jobs_ratio": delayed_jobs_ratio,
        "total_job_count": total_jobs
    }


def _read_parallelism_map(parallelism_file: str) -> Optional[Dict[str, int]]:
    """
    Reads a CSV file and returns a dictionary mapping dates to worker counts.
    """
    if not os.path.exists(parallelism_file):
        print(f"Warning: Parallelism file not found at '{parallelism_file}'.")
        return None

    parallelism_map = {}
    try:
        with open(parallelism_file, mode='r', newline='') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                parallelism_map[row['date']] = int(row['degree_of_parallelism'])
        return parallelism_map
    except (IOError, csv.Error, ValueError) as e:
        print(f"Error reading or parsing parallelism file: {e}")
        return None


def analyze_job_queue(
    job_data: List[Tuple[str, str, str]],
    num_workers: int,
    timestamp_format: str = "%Y-%m-%dT%H:%M:%S",
    parallelism_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyzes a job queueing system with timestamp data and calculates waiting time statistics.
    """
    if not job_data:
        return _calculate_metrics([], 0, 0)

    try:
        jobs = _prepare_jobs(job_data, timestamp_format)
    except ValueError:
        return {}

    if not jobs:
        return _calculate_metrics([], 0, 0)

    parallelism_map = None
    if parallelism_file:
        parallelism_map = _read_parallelism_map(parallelism_file)

    initial_date = datetime.fromtimestamp(jobs[0].start_time).strftime('%Y-%m-%d')
    initial_workers = num_workers
    if parallelism_map and initial_date in parallelism_map:
        initial_workers = parallelism_map[initial_date]

    if initial_workers <= 0:
        return _calculate_metrics([], 0, 0)

    wait_times, delayed_job_count = _run_simulation(jobs, initial_workers, parallelism_map)

    return _calculate_metrics(wait_times, len(jobs), delayed_job_count)


# --- Main execution block ---
if __name__ == "__main__":
    # Example Data (Timestamp format)
    sample_job_data_timestamp = [
        ('job_1', '2025-09-26T10:00:00', '2025-09-26T10:00:05'),  # duration: 5s
        ('job_2', '2025-09-26T10:00:01', '2025-09-26T10:00:04'),  # duration: 3s
        ('job_3', '2025-09-26T10:00:02', '2025-09-26T10:00:08'),  # duration: 6s
        ('job_4', '2025-09-26T10:00:03', '2025-09-26T10:00:07'),  # duration: 4s
        ('job_5', '2025-09-26T10:00:06', '2025-09-26T10:00:10'),  # duration: 4s
        ('job_6', '2025-09-26T10:00:08', '2025-09-26T10:00:12'),  # duration: 4s
        ('job_7', '2025-09-26T10:00:09', '2025-09-26T10:00:11'),  # duration: 2s
        ('job_8', '2025-09-26T10:00:13', '2025-09-26T10:00:15'),  # duration: 2s
    ]

    # Simulation Parameters
    concurrent_workers = 2
    ts_format = "%Y-%m-%dT%H:%M:%S"

    print(f"Starting simulation with {concurrent_workers} concurrent workers...")
    print("-" * 50)

    # Run the analyzer
    results = analyze_job_queue(
        sample_job_data_timestamp,
        concurrent_workers,
        timestamp_format=ts_format
    )

    # Print results
    if results:
        print("Simulation Performance Analysis:")
        print(f"  - Total Jobs Analyzed : {results['total_job_count']}")
        print(f"  - Delayed Jobs        : {results['delayed_job_count']} ({results.get('delayed_jobs_ratio', 0.0):.2%})")
        print("-" * 25)
        print(f"  - Total Wait Time     : {results['total_wait_time']:.2f} seconds")
        print(f"  - Average Wait Time   : {results['average_wait_time']:.2f} seconds")
        print(f"  - Wait Time Variance  : {results['variance']:.2f}")
        print(f"  - Wait Time Std Dev   : {results['std_deviation']:.2f}")
        # Example of how to use the wait times list:
        # wait_times_str = [f'{t:.2f}' for t in results.get('wait_times_list', [])]
        # print(f"  - All Wait Times (s)  : {wait_times_str}")
        print("-" * 50)