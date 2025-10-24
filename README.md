# Job Queue Analyzer

This project simulates a job queueing system and analyzes its performance. It includes a command-line interface for running simulations and a FastAPI server for serving analysis results.

## Features

- Simulates a job queue with a configurable number of workers.
- Calculates wait time statistics, including total wait time, average wait time, and variance.
- Provides a FastAPI server with JWT authentication to serve analysis results.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command-Line Interface

To run a simulation from the command line, you can execute the `DelayedRequest.py` script:
```bash
python3 DelayedRequest.py
```

### FastAPI Server

To start the FastAPI server, run the following command:
```bash
uvicorn main:app --reload
```

The server will be available at `http://127.0.0.1:8000`.

#### API Endpoints

- `POST /token`: Authenticate and receive a JWT.
- `GET /users/me`: Get information about the current user (requires authentication).
- `POST /analyze`: Analyze a job queue (requires authentication).
