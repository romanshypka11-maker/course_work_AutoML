from state import AutoMLState


def _quality_is_acceptable(metrics: dict[str, float]) -> bool:
    """Use conservative, explainable thresholds only when a comparable metric exists."""
    for name in ("accuracy", "f1", "f1_macro", "roc_auc"):
        if name in metrics:
            return metrics[name] >= 0.50
    if "r2" in metrics:
        return metrics["r2"] >= 0.0
    # A valid metric report is still a successful technical run. The UI lets the
    # user assess task-specific metrics that do not have a universal threshold.
    return bool(metrics)


def evaluator_node(state: AutoMLState) -> dict:
    print("--- [4/4] EVALUATOR: validating execution and metrics ---")
    attempt = state.get("iteration_count", 0)
    if not state.get("last_success", False):
        stderr = (state.get("last_stderr") or "No error output was returned.")[-1800:]
        feedback = (
            f"Attempt {attempt} failed during execution. Fix the generated script using this traceback:\n{stderr}"
        )
        return {
            "feedback": feedback,
            "evaluation_report": "The generated script did not complete successfully.",
            "status": "retrying" if state.get("error_type") != "fatal" else "failed",
        }

    metrics = state.get("last_metrics") or {}
    if not metrics:
        return {
            "feedback": (
                "The script completed but out/metrics.json did not contain a JSON object with numeric metrics. "
                "Write valid numeric metrics and print them."
            ),
            "error_type": "runtime_error",
            "evaluation_report": "No usable numeric metrics were produced.",
            "status": "retrying",
        }

    if _quality_is_acceptable(metrics):
        return {
            "feedback": None,
            "error_type": None,
            "evaluation_report": f"Pipeline completed successfully. Metrics: {metrics}",
            "status": "success",
        }

    return {
        "feedback": (
            f"Metrics from attempt {attempt} are below the baseline: {metrics}. "
            "Revise preprocessing, validation, model choice or class balancing and generate another pipeline."
        ),
        "error_type": "fixable_metric",
        "evaluation_report": f"Pipeline executed, but quality should be improved. Metrics: {metrics}",
        "status": "bad_metrics",
    }
