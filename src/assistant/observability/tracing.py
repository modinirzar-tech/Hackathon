"""Lightweight W3C-traceparent tracer + metrics registry. Spans are kept per trace and aggregated per
name for KPI percentiles; export to an OTLP collector is a thin adapter (see otel_exporter plugin)."""
from __future__ import annotations

import contextvars
import json
import os
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field


def new_trace_id() -> str:
    return os.urandom(16).hex()


def new_span_id() -> str:
    return os.urandom(8).hex()


def make_traceparent(trace_id: str | None = None) -> str:
    return f"00-{trace_id or new_trace_id()}-{new_span_id()}-01"


def percentile(values, q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, max(0, int(round(q / 100 * (len(s) - 1)))))]


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    start: float
    duration_ms: float = 0.0
    attrs: dict = field(default_factory=dict)


class Metrics:
    def __init__(self, window: int = 5000):
        self.counters: dict[str, float] = defaultdict(float)
        self.hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))

    def inc(self, name: str, v: float = 1.0):
        self.counters[name] += v

    def observe(self, name: str, v: float):
        self.hist[name].append(v)

    def p(self, name: str, q: float) -> float:
        return percentile(self.hist.get(name, ()), q)

    def prometheus(self) -> str:
        lines = []
        for k, v in sorted(self.counters.items()):
            lines.append(f"assistant_{k}_total {v}")
        for k, vals in sorted(self.hist.items()):
            key = k.replace(".", "_")
            for q in (50, 95, 99):
                lines.append(f'assistant_{key}_ms{{quantile="0.{q}"}} {percentile(vals, q):.3f}')
        return "\n".join(lines) + "\n"


_current_trace: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_id", default=None)


class Tracer:
    def __init__(self, metrics: Metrics, export_path: str | None = None, keep: int = 200):
        self.metrics = metrics
        self.traces: dict[str, list[Span]] = {}
        self._order: deque[str] = deque(maxlen=keep)
        self.export_path = export_path or os.environ.get("OTEL_JSONL_EXPORT")
        self.exporters: list = []

    def start_trace(self, traceparent: str | None = None) -> str:
        tid = traceparent.split("-")[1] if traceparent else new_trace_id()
        _current_trace.set(tid)
        if tid not in self.traces:
            if len(self._order) == self._order.maxlen:
                self.traces.pop(self._order[0], None)
            self._order.append(tid)
            self.traces[tid] = []
        return tid

    @contextmanager
    def span(self, name: str, **attrs):
        tid = _current_trace.get() or self.start_trace()
        sp = Span(name, tid, new_span_id(), time.perf_counter(), attrs=dict(attrs))
        try:
            yield sp
        finally:
            sp.duration_ms = (time.perf_counter() - sp.start) * 1000
            self.traces.setdefault(tid, []).append(sp)
            self.metrics.observe(name, sp.duration_ms)
            for ex in self.exporters:
                try:
                    ex(sp)
                except Exception:
                    self.metrics.inc("otel_export_errors")
            if self.export_path:
                with open(self.export_path, "a") as fh:
                    fh.write(json.dumps({"name": name, "trace_id": tid, "span_id": sp.span_id,
                                         "duration_ms": round(sp.duration_ms, 3), "attrs": sp.attrs}, default=str) + "\n")

    def get(self, trace_id: str) -> list[Span]:
        return self.traces.get(trace_id, [])
