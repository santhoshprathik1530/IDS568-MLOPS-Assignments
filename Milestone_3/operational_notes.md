# Operational Notes

## Retry Strategy and Failure Handling
1. DAG-level retries:
   - `retries=2`
   - `retry_delay=timedelta(minutes=2)`
2. Failure callback:
   - `on_failure_callback` logs failed task ID, try number, and run context.
3. Failure test hook:
   - Set `M3_FAIL_ONCE_TRAIN=true` to intentionally fail first `train_model` attempt and verify retry behavior.

## Monitoring and Alerting Recommendations
1. Pipeline reliability:
   - Track Airflow DAG run success rate, task retry count, and task duration.
2. Model quality:
   - Alert when `accuracy < 0.90` or `f1_score < 0.90` (same thresholds as CI quality gate).
3. Data/lineage integrity:
   - Alert if model artifact hash tags are missing (`model_sha256`, `model_md5`).
   - Track drift in feature distributions for real datasets.
4. Registry governance:
   - Require review before transitioning to `Production`.

## Rollback Procedures
1. Identify prior stable model version in MLflow registry (`milestone3-iris-model`).
2. Transition current `Production` to `Archived` (or demote to `Staging`).
3. Promote previous stable version to `Production`.
4. Record rollback reason in model version description and tags (`rollback_reason`, `rollback_timestamp`).
5. Re-run validation gate on rollback candidate before finalizing.

## Runbook Commands
```bash
# Validate latest metrics output
python model_validation.py --metrics-file outputs/metrics_latest.json --min-accuracy 0.90 --min-f1 0.90

# Run a deterministic training candidate
python train.py --run-name rollback-check --data-version rollback_v1 --n-estimators 100 --max-depth 5
```
