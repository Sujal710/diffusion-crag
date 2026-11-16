"""Optional MLflow tracing of the CRAG pipeline.

With MLFLOW_TRACKING_URI set, every question becomes one trace in MLflow: each LangGraph step
(retrieve, evaluate_documents, web_search, generate, check_answer, ...) and each LLM call inside
it is recorded with its inputs, outputs and timing. Without it, nothing here does anything.
Tracing never changes which models are used, and a tracing failure never breaks a question.
"""
import logging

from .config import get_settings

log = logging.getLogger("crag.tracing")
_enabled = False


def setup_tracing() -> bool:
    """Turn on MLflow auto-tracing for LangChain / LangGraph. Safe to call more than once."""
    global _enabled
    s = get_settings()
    if _enabled or not s.mlflow_tracking_uri:
        return _enabled
    try:
        import mlflow
        import mlflow.langchain

        mlflow.set_tracking_uri(s.mlflow_tracking_uri)
        mlflow.set_experiment(s.mlflow_experiment)
        mlflow.langchain.autolog()
        _enabled = True
        log.warning("MLflow tracing on: %s (experiment %r)", s.mlflow_tracking_uri, s.mlflow_experiment)
    except Exception as e:  # tracing is optional: never stop the app because of it
        log.warning("MLflow tracing could not be started (%s); continuing without it.", e)
    return _enabled
