import heapq
import numpy as np
from collections import namedtuple
from datetime import datetime

# A simple data structure for a job
# start_time and end_time will be stored as numeric Unix timestamps (seconds since epoch)
Job = namedtuple('Job', ['uid', 'start_time', 'end_time', 'duration'])

def simulate_job_queue_timestamp(job_data, num_workers, timestamp_format="%Y-%m-%dT%H:%M:%S"):
    """
    Simulates a job queueing system with timestamp data and calculates waiting time statistics,
    including the count of delayed jobs.

    Args:
        job_data (list): A list of tuples, where each tuple represents a job
                         as (uid, start_time_str, end_time_str).
        num_workers (int): The number of concurrent workers available to process jobs.
        timestamp_format (str): The format of the timestamp strings in the input data.

    Returns:
        dict: A dictionary containing the calculated performance metrics.
    """
    if not job_data or num_workers <= 0:
        return {
            "total_wait_time": 0,
            "average_wait_time": 0,
            "variance": 0,
            "std_deviation": 0,
            "wait_times_list": [],
            "delayed_job_count": 0,
            "total_job_count": 0
        }

    # 1. Prepare jobs: parse timestamps, convert to numeric values, and calculate duration
    jobs = []
    try:
        for uid, start_str, end_str in job_data:
            start_dt = datetime.strptime(start_str, timestamp_format)
            end_dt = datetime.strptime(end_str, timestamp_format)
            start_ts = start_dt.timestamp()
            end_ts = end_dt.timestamp()

            if start_ts > end_ts:
                print(f"Warning: Job {uid} has start_time > end_time. Skipping.")
                continue
            
            duration = end_ts - start_ts
            jobs.append(Job(uid, start_ts, end_ts, duration))
    except ValueError as e:
        print(f"Error parsing timestamp. Please check the timestamp_format.")
        print(f"Details: {e}")
        return {}
    
    jobs.sort(key=lambda x: x.start_time)

    # 2. Initialize simulation variables
    first_job_start_time = jobs[0].start_time
    worker_finish_times = [first_job_start_time] * num_workers
    heapq.heapify(worker_finish_times)
    
    wait_times = [] 
    delayed_job_count = 0 # 지연된 작업 수를 세기 위한 카운터

    # 3. Run the simulation
    for job in jobs:
        earliest_worker_free_time = heapq.heappop(worker_finish_times)
        actual_start_time = max(job.start_time, earliest_worker_free_time)
        
        wait_time = actual_start_time - job.start_time
        wait_times.append(wait_time)

        # 대기 시간이 0보다 크면 지연된 작업으로 간주하고 카운트를 1 증가
        # 부동 소수점 오차를 고려하여 아주 작은 값(epsilon)보다 큰지 확인
        if wait_time > 1e-9:
            delayed_job_count += 1

        finish_time = actual_start_time + job.duration
        heapq.heappush(worker_finish_times, finish_time)

    # 4. Calculate performance metrics
    total_wait_time = np.sum(wait_times)
    average_wait_time = np.mean(wait_times)
    variance = np.var(wait_times)
    std_deviation = np.std(wait_times)

    return {
        "total_wait_time": total_wait_time,
        "average_wait_time": average_wait_time,
        "variance": variance,
        "std_deviation": std_deviation,
        "delayed_job_count": delayed_job_count,
        "total_job_count": len(jobs)
    }

# Main execution block
if __name__ == "__main__":
    # --- 예시 데이터 (타임스탬프 형식) ---
    sample_job_data_timestamp = [
        ('job_1', '2025-09-26T10:00:00', '2025-09-26T10:00:05'), # duration: 5s
        ('job_2', '2025-09-26T10:00:01', '2025-09-26T10:00:04'), # duration: 3s
        ('job_3', '2025-09-26T10:00:02', '2025-09-26T10:00:08'), # duration: 6s
        ('job_4', '2025-09-26T10:00:03', '2025-09-26T10:00:07'), # duration: 4s
        ('job_5', '2025-09-26T10:00:06', '2025-09-26T10:00:10'), # duration: 4s
        ('job_6', '2025-09-26T10:00:08', '2025-09-26T10:00:12'), # duration: 4s
        ('job_7', '2025-09-26T10:00:09', '2025-09-26T10:00:11'), # duration: 2s
        ('job_8', '2025-09-26T10:00:13', '2025-09-26T10:00:15'), # duration: 2s
    ]

    # --- 시뮬레이션 파라미터 ---
    concurrent_workers = 2 
    ts_format = "%Y-%m-%dT%H:%M:%S"

    print(f"'{concurrent_workers}'개의 동시 작업으로 시뮬레이션을 시작합니다...")
    print("-" * 50)

    # 시뮬레이터 실행
    results = simulate_job_queue_timestamp(
        sample_job_data_timestamp, 
        concurrent_workers,
        timestamp_format=ts_format
    )

    # 결과 출력
    if results:
        print("시뮬레이션 성능 분석 결과:")
        print(f"  - 총 작업 수          : {results['total_job_count']} 개")
        print(f"  - 지연 시작된 작업 수 : {results['delayed_job_count']} 개")
        print("-" * 25)
        print(f"  - 총 대기 시간        : {results['total_wait_time']:.2f} 초")
        print(f"  - 평균 대기 시간      : {results['average_wait_time']:.2f} 초")
        print(f"  - 대기 시간 분산      : {results['variance']:.2f}")
        print(f"  - 대기 시간 표준편차  : {results['std_deviation']:.2f}")
        print("-" * 50)

