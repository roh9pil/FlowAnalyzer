import os
from datetime import datetime, timedelta, timezone
import pandas as pd
from DelayedRequest import analyze_job_queue

from sqlalchemy import create_engine, text, MetaData, Table
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError

# -----------------------------
# 1) Config
# -----------------------------
PG_URI = os.getenv("PG_URI")  # postgresql+psycopg2://user:pw@host:5432/db

SOURCE_SCHEMA = "public"
SOURCE_TABLE = "review_history"
SOURCE_DATE_COL = "created_on"

TARGET_SCHEMA = "analysis"
TARGET_TABLE = "review_analyzed"

UPSERT_KEY = "uid"  # ✅ 확정

# -----------------------------
# 2) Predict wait time function
# -----------------------------


def predict_wait_time(df: pd.DataFrame, num_workers: int) -> pd.DataFrame:
    """
    Analyzes job queue performance and adds the average wait time to the DataFrame.

    Assumes the input DataFrame contains the following columns:
    - uid: A unique identifier for the job.
    - created_on: The submission time of the job (datetime object).
    - updated_on: The completion time of the job (datetime object).
    """
    df = df.copy()

    if df.empty:
        df["avg_wait_time"] = None
        return df

    # 1. Transform data for analyze_job_queue
    # Expected format: List[Tuple[str, str, str]]
    # (uid, start_time_str, end_time_str)
    job_data = [
        (
            row.uid,
            row.created_on.strftime("%Y-%m-%dT%H:%M:%S"),
            row.updated_on.strftime("%Y-%m-%dT%H:%M:%S")
        )
        for row in df.itertuples(index=False)
        if hasattr(row, 'uid') and hasattr(row, 'created_on') and hasattr(row, 'updated_on')
    ]

    # 2. Call the analysis function
    results = analyze_job_queue(
        job_data,
        num_workers=num_workers,
        timestamp_format="%Y-%m-%dT%H:%M:%S"
    )

    # 3. Add results to the DataFrame
    avg_wait_time = results.get("average_wait_time")
    df["avg_wait_time"] = avg_wait_time

    return df

# -----------------------------
# 3) Calculate yesterday range (KST)
# -----------------------------
def get_yesterday_range_kst():
    KST = timezone(timedelta(hours=9))
    now_kst = datetime.now(KST)

    today_start = now_kst.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_start = today_start - timedelta(days=1)

    # created_on 컬럼이 timestamp without tz인 경우에 맞춰 naive로 전달
    return yesterday_start.replace(tzinfo=None), today_start.replace(tzinfo=None)

# -----------------------------
# 4) Extract yesterday data
# -----------------------------
def extract_yesterday_data(engine):
    y_start, y_end = get_yesterday_range_kst()

    query = f"""
        SELECT *
        FROM {SOURCE_SCHEMA}.{SOURCE_TABLE}
        WHERE {SOURCE_DATE_COL} >= :start_dt
          AND {SOURCE_DATE_COL} < :end_dt
    """

    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params={"start_dt": y_start, "end_dt": y_end})

    return df

# -----------------------------
# 5) Ensure schema/table + constraints
# -----------------------------
def ensure_target_ready(engine, df):
    """
    - schema 존재 보장
    - target table이 없으면 df schema 기반으로 생성
    - UPSERT_KEY (uid)에 UNIQUE constraint 없으면 추가
    """
    if df.empty:
        return

    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA};"))

        # target table 존재 여부
        exists_q = """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = :schema AND table_name = :table
        )
        """
        exists = conn.execute(text(exists_q), {"schema": TARGET_SCHEMA, "table": TARGET_TABLE}).scalar()

        if not exists:
            print("[WARN] Target table does not exist -> creating based on extracted columns...")
            df.head(0).to_sql(
                TARGET_TABLE,
                con=engine,
                schema=TARGET_SCHEMA,
                if_exists="replace",
                index=False
            )

        # uid 컬럼 존재 여부
        col_check_q = """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = :schema
              AND table_name = :table
              AND column_name = :col
        )
        """
        key_exists = conn.execute(
            text(col_check_q),
            {"schema": TARGET_SCHEMA, "table": TARGET_TABLE, "col": UPSERT_KEY}
        ).scalar()

        if not key_exists:
            raise ValueError(f"[ERROR] UPSERT_KEY '{UPSERT_KEY}' does not exist in target table.")

        # UNIQUE constraint 존재 여부
        constraint_name = f"{TARGET_TABLE}_{UPSERT_KEY}_uniq"
        constraint_exists_q = """
        SELECT EXISTS (
            SELECT 1
            FROM pg_constraint c
            JOIN pg_class t ON c.conrelid = t.oid
            JOIN pg_namespace n ON n.oid = t.relnamespace
            WHERE n.nspname = :schema
              AND t.relname = :table
              AND c.conname = :constraint_name
        )
        """
        constraint_exists = conn.execute(
            text(constraint_exists_q),
            {"schema": TARGET_SCHEMA, "table": TARGET_TABLE, "constraint_name": constraint_name}
        ).scalar()

        if not constraint_exists:
            print(f"[INFO] Creating UNIQUE constraint on {UPSERT_KEY}...")
            conn.execute(
                text(f"""
                    ALTER TABLE {TARGET_SCHEMA}.{TARGET_TABLE}
                    ADD CONSTRAINT {constraint_name} UNIQUE ({UPSERT_KEY});
                """)
            )

# -----------------------------
# 6) Upsert using SQLAlchemy Core (Postgres dialect)
# -----------------------------
def upsert_to_target(engine, df: pd.DataFrame):
    if df.empty:
        print("[INFO] No data for yesterday. Nothing to load.")
        return

    metadata = MetaData(schema=TARGET_SCHEMA)
    target = Table(TARGET_TABLE, metadata, autoload_with=engine)

    # dict records
    records = df.to_dict(orient="records")

    stmt = insert(target).values(records)

    # update set clause (exclude UPSERT_KEY)
    update_cols = {
        c.name: getattr(stmt.excluded, c.name)
        for c in target.columns
        if c.name != UPSERT_KEY
    }

    stmt = stmt.on_conflict_do_update(
        index_elements=[UPSERT_KEY],
        set_=update_cols
    )

    with engine.begin() as conn:
        conn.execute(stmt)

    print(f"[INFO] Upserted {len(df)} rows into {TARGET_SCHEMA}.{TARGET_TABLE}")

# -----------------------------
# 7) Main job
# -----------------------------
def main():
    if not PG_URI:
        raise ValueError("PG_URI environment variable is required. ex) postgresql+psycopg2://...")

    engine = create_engine(PG_URI)

    try:
        print("[INFO] Extracting yesterday data...")
        df = extract_yesterday_data(engine)
        print(f"[INFO] Extracted {len(df)} rows.")

        if df.empty:
            print("[SUCCESS] No data for yesterday. Job ended.")
            return

        print("[INFO] Applying transformation (dummy predict=1)...")
        df = predict_wait_time(df, num_workers=2)

        print("[INFO] Ensuring target schema/table/constraints...")
        ensure_target_ready(engine, df)

        print("[INFO] Loading via UPSERT (uid)...")
        upsert_to_target(engine, df)

        print("[SUCCESS] Job completed successfully.")

    except (SQLAlchemyError, ValueError) as e:
        print(f"[FAILED] Job failed due to error: {e}")
        raise

if __name__ == "__main__":
    main()