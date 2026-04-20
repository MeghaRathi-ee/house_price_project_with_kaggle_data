from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.dates import days_ago
import json
import os


# ─────────────────────────────────────────────
# WHAT IS AIRFLOW?
#
# Airflow is a pipeline orchestrator.
# You define a DAG (Directed Acyclic Graph)
# which is just a sequence of tasks with
# dependencies between them.
#
# WHY USE IT?
# Without Airflow: you manually run dvc repro
# With Airflow: pipeline runs automatically
# on a schedule — weekly, daily, or on trigger
#
# HOW IT WORKS:
# 1. You write a DAG file (this file)
# 2. Airflow scheduler reads it
# 3. At the scheduled time, it runs tasks
#    in the correct order
# 4. If a task fails, it retries automatically
# 5. You see all runs in the Airflow UI
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# Project path — adjust to your local path
# ─────────────────────────────────────────────
PROJECT_DIR = os.path.expanduser("~/Desktop/hpp_12_04_26")
PYTHON_BIN  = "python"


# ─────────────────────────────────────────────
# Default arguments for all tasks
# ─────────────────────────────────────────────
default_args = {
    "owner"           : "megha",
    "retries"         : 2,                        # retry failed tasks 2 times
    "retry_delay"     : timedelta(minutes=5),     # wait 5 min before retry
    "email_on_failure": False,
    "email_on_retry"  : False,
    "depends_on_past" : False,                    # don't wait for previous run
}


# ─────────────────────────────────────────────
# DAG definition
# schedule_interval="@weekly" = runs every Monday 00:00
# catchup=False = don't run missed runs
# ─────────────────────────────────────────────
with DAG(
    dag_id          = "house_price_ml_pipeline",
    description     = "Weekly house price model retraining pipeline",
    default_args    = default_args,
    start_date      = days_ago(1),
    schedule_interval = "@weekly",
    catchup         = False,
    tags            = ["mlops", "house_price", "weekly"],
) as dag:

    # ──────────────────────────────────────
    # TASK 1: Ingest data
    # Loads raw CSV, splits train/test
    # ──────────────────────────────────────
    ingest = BashOperator(
        task_id         = "ingest",
        bash_command    = f"cd {PROJECT_DIR} && {PYTHON_BIN} src/ingest.py",
    )

    # ──────────────────────────────────────
    # TASK 2: Preprocess
    # Feature engineering, scaling, encoding
    # ──────────────────────────────────────
    preprocess = BashOperator(
        task_id         = "preprocess",
        bash_command    = f"cd {PROJECT_DIR} && {PYTHON_BIN} src/preprocess.py",
    )

    # ──────────────────────────────────────
    # TASK 3: Train
    # Train 8 models, log to MLFlow
    # ──────────────────────────────────────
    train = BashOperator(
        task_id         = "train",
        bash_command    = f"cd {PROJECT_DIR} && {PYTHON_BIN} src/train.py",
    )

    # ──────────────────────────────────────
    # TASK 4: Evaluate
    # Compute metrics, save to JSON
    # ──────────────────────────────────────
    evaluate = BashOperator(
        task_id         = "evaluate",
        bash_command    = f"cd {PROJECT_DIR} && {PYTHON_BIN} src/evaluate.py",
    )

    # ──────────────────────────────────────
    # TASK 5: Monitor
    # Run Evidently drift report
    # ──────────────────────────────────────
    monitor = BashOperator(
        task_id         = "monitor",
        bash_command    = f"cd {PROJECT_DIR} && {PYTHON_BIN} src/monitor.py",
    )

    # ──────────────────────────────────────
    # TASK 6: Check retrain trigger
    # Read retrain_trigger.json
    # Branch: retrain or skip
    # ──────────────────────────────────────
    def check_retrain_trigger(**context):
        """
        BranchPythonOperator — returns task_id of next task to run.
        If drift detected → retrain (run ingest again)
        If no drift → notify_no_retrain
        """
        trigger_path = os.path.join(
            PROJECT_DIR, "data/reports/retrain_trigger.json"
        )
        with open(trigger_path) as f:
            trigger = json.load(f)

        should_retrain = trigger.get("should_retrain", False)
        reasons        = trigger.get("reason", [])

        print(f"[airflow] should_retrain: {should_retrain}")
        print(f"[airflow] reasons: {reasons}")

        if should_retrain:
            print("[airflow] Drift detected → triggering retrain")
            return "retrain_notify"
        else:
            print("[airflow] No drift → skipping retrain")
            return "no_retrain_notify"

    check_trigger = BranchPythonOperator(
        task_id         = "check_retrain_trigger",
        python_callable = check_retrain_trigger,
        provide_context = True,
    )

    # ──────────────────────────────────────
    # TASK 7a: Retrain notification
    # In production: send Slack/email alert
    # ──────────────────────────────────────
    retrain_notify = BashOperator(
        task_id      = "retrain_notify",
        bash_command = (
            "echo '⚠️  DRIFT DETECTED — Retraining triggered' && "
            "echo 'In production: this would send a Slack message' && "
            "echo 'and trigger a new dvc repro run'"
        ),
    )

    # ──────────────────────────────────────
    # TASK 7b: No retrain notification
    # ──────────────────────────────────────
    no_retrain_notify = BashOperator(
        task_id      = "no_retrain_notify",
        bash_command = "echo '✅  No drift detected — model is healthy'",
    )

    # ──────────────────────────────────────
    # TASK 8: Push to DVC remote
    # Save updated artifacts to Azure
    # ──────────────────────────────────────
    dvc_push = BashOperator(
        task_id      = "dvc_push",
        bash_command = f"cd {PROJECT_DIR} && dvc push",
        trigger_rule = "none_failed_min_one_success",
    )

    # ──────────────────────────────────────
    # TASK DEPENDENCIES
    # Defines the order of execution
    # >> means "runs before"
    # ──────────────────────────────────────
    ingest >> preprocess >> train >> evaluate >> monitor >> check_trigger
    check_trigger >> [retrain_notify, no_retrain_notify]
    [retrain_notify, no_retrain_notify] >> dvc_push