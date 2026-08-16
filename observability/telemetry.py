from __future__ import annotations

import hashlib
import json
import os
import statistics
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENT_PATH = ROOT / "artifacts" / "observability" / "events.jsonl"
_WRITE_LOCK = threading.Lock()
_METRICS_SERVER_STARTED = False

try:
    from prometheus_client import Counter, Histogram, start_http_server

    QUERY_TOTAL = Counter(
        "kitti_query_total",
        "Number of KITTI Explorer queries",
        ["mode", "route", "status"],
    )
    QUERY_LATENCY = Histogram(
        "kitti_query_latency_seconds",
        "End-to-end query latency",
        ["mode", "route"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
    )
    RESULT_COUNT = Histogram(
        "kitti_query_result_count",
        "Number of frames returned by a query",
        ["mode", "route"],
        buckets=(0, 1, 3, 5, 10, 25, 50, 100, 500),
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in minimal installations
    PROMETHEUS_AVAILABLE = False


def _event_path() -> Path:
    configured = os.getenv("KITTI_OBSERVABILITY_PATH")
    return Path(configured) if configured else DEFAULT_EVENT_PATH


def _query_fingerprint(query: str | None) -> str | None:
    if not query:
        return None
    return hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()[:16]


def record_event(
    event_type: str,
    *,
    query: str | None = None,
    mode: str = "unknown",
    route: str = "unknown",
    status: str = "success",
    latency_ms: float | None = None,
    result_count: int | None = None,
    fallback_used: bool = False,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append a structured event and update Prometheus metrics.

    Raw user queries are not persisted. A stable fingerprint allows repeated-query
    analysis without storing potentially sensitive text.
    """
    event = {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "query_id": _query_fingerprint(query),
        "mode": mode,
        "route": route,
        "status": status,
        "latency_ms": round(float(latency_ms), 3) if latency_ms is not None else None,
        "result_count": int(result_count) if result_count is not None else None,
        "fallback_used": bool(fallback_used),
        "metadata": metadata or {},
    }

    path = _event_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")

    if PROMETHEUS_AVAILABLE and event_type == "query":
        labels = {"mode": mode, "route": route, "status": status}
        QUERY_TOTAL.labels(**labels).inc()
        if latency_ms is not None:
            QUERY_LATENCY.labels(mode=mode, route=route).observe(latency_ms / 1000.0)
        if result_count is not None:
            RESULT_COUNT.labels(mode=mode, route=route).observe(result_count)
    return event


def iter_events(limit: int = 1000) -> Iterable[dict[str, Any]]:
    """Yield the most recent valid telemetry events."""
    path = _event_path()
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    events = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def get_dashboard_summary(limit: int = 1000) -> dict[str, Any]:
    """Aggregate recent query events for the Streamlit health panel."""
    events = [e for e in iter_events(limit) if e.get("event_type") == "query"]
    latencies = [float(e["latency_ms"]) for e in events if e.get("latency_ms") is not None]
    successes = sum(e.get("status") == "success" for e in events)
    fallbacks = sum(bool(e.get("fallback_used")) for e in events)
    return {
        "queries": len(events),
        "success_rate": successes / len(events) if events else None,
        "fallback_rate": fallbacks / len(events) if events else None,
        "latency_mean_ms": statistics.fmean(latencies) if latencies else None,
        "latency_p50_ms": _percentile(latencies, 0.50),
        "latency_p95_ms": _percentile(latencies, 0.95),
    }


def start_metrics_server(port: int | None = None) -> bool:
    """Start the optional Prometheus endpoint once per process."""
    global _METRICS_SERVER_STARTED
    if not PROMETHEUS_AVAILABLE or _METRICS_SERVER_STARTED:
        return False
    resolved_port = port or int(os.getenv("KITTI_METRICS_PORT", "8001"))
    start_http_server(resolved_port)
    _METRICS_SERVER_STARTED = True
    return True
