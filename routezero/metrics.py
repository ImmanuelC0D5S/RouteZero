from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

try:
    import aiosqlite
except ImportError:
    aiosqlite = None

from routezero.config import Settings


@dataclass
class RequestLogRecord:
    request_id: str
    cache_status: str
    selected_route: str
    task_type: str
    saved_tokens: int
    latency_ms: float
    verification_passed: bool | None
    timestamp: datetime


@dataclass
class DashboardSnapshot:
    cache_hit_ratio: float
    local_offload_rate: float
    remote_fallback_rate: float
    total_token_cost_savings: int
    avg_local_latency_ms: float
    avg_remote_latency_ms: float
    verification_success_rate: float
    routing_breakdown: dict[str, int]


class _FallbackStore:
    """In-memory list fallback when aiosqlite is unavailable."""

    def __init__(self) -> None:
        self.records: list[RequestLogRecord] = []
        self._lock = asyncio.Lock()

    async def init(self) -> None:
        pass

    async def insert_many(self, records: list[RequestLogRecord]) -> None:
        async with self._lock:
            self.records.extend(records)

    async def fetch_all(self) -> list[RequestLogRecord]:
        async with self._lock:
            return list(self.records)


class MetricsLogger:
    def __init__(self, settings: Settings) -> None:
        self.db_path = settings.sqlite_path
        self._queue: asyncio.Queue[RequestLogRecord] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._batch: list[RequestLogRecord] = []
        self._batch_event = asyncio.Event()
        self._in_memory: _FallbackStore | None = None

    async def init_db(self) -> None:
        if aiosqlite is not None:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """CREATE TABLE IF NOT EXISTS requests(
                        id TEXT PRIMARY KEY,
                        request_id TEXT,
                        cache_status TEXT,
                        selected_route TEXT,
                        task_type TEXT,
                        saved_tokens INTEGER,
                        latency_ms REAL,
                        verification_passed INTEGER,
                        timestamp TEXT
                    )"""
                )
                await db.commit()
        else:
            self._in_memory = _FallbackStore()
            await self._in_memory.init()

        self._worker_task = asyncio.create_task(self._worker())

    async def _worker(self) -> None:
        while True:
            try:
                timeout = 5.0
                with_timeout = asyncio.create_task(
                    asyncio.wait_for(self._queue.get(), timeout=timeout)
                )
                try:
                    record = await with_timeout
                    self._batch.append(record)
                    self._queue.task_done()
                except asyncio.TimeoutError:
                    pass

                if len(self._batch) >= 10:
                    await self._flush()

                self._batch_event.set()
            except asyncio.CancelledError:
                await self._flush()
                break

    async def _flush(self) -> None:
        if not self._batch:
            return
        to_insert = self._batch[:]
        self._batch.clear()

        if self._in_memory is not None:
            await self._in_memory.insert_many(to_insert)
        elif aiosqlite is not None:
            try:
                async with aiosqlite.connect(self.db_path) as db:
                    rows = [
                        (
                            str(uuid.uuid4()),
                            r.request_id,
                            r.cache_status,
                            r.selected_route,
                            r.task_type,
                            r.saved_tokens,
                            r.latency_ms,
                            1 if r.verification_passed is True else 0 if r.verification_passed is False else None,
                            r.timestamp.isoformat(),
                        )
                        for r in to_insert
                    ]
                    await db.executemany(
                        """INSERT INTO requests
                        (id, request_id, cache_status, selected_route, task_type,
                         saved_tokens, latency_ms, verification_passed, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        rows,
                    )
                    await db.commit()
            except Exception:
                pass

    async def log_request(self, record: RequestLogRecord) -> None:
        await self._queue.put(record)

    async def get_dashboard_snapshot(self) -> DashboardSnapshot:
        if self._in_memory is not None:
            records = await self._in_memory.fetch_all()
        elif aiosqlite is not None:
            try:
                async with aiosqlite.connect(self.db_path) as db:
                    db.row_factory = aiosqlite.Row
                    cursor = await db.execute("SELECT * FROM requests")
                    rows = await cursor.fetchall()
                    records = [
                        RequestLogRecord(
                            request_id=r["request_id"],
                            cache_status=r["cache_status"],
                            selected_route=r["selected_route"],
                            task_type=r["task_type"],
                            saved_tokens=r["saved_tokens"],
                            latency_ms=r["latency_ms"],
                            verification_passed={
                                1: True,
                                0: False,
                                None: None,
                            }.get(r["verification_passed"]),
                            timestamp=datetime.fromisoformat(r["timestamp"]),
                        )
                        for r in rows
                    ]
            except Exception:
                records = []
        else:
            records = []

        total = len(records)
        if total == 0:
            return DashboardSnapshot(
                cache_hit_ratio=0.0,
                local_offload_rate=0.0,
                remote_fallback_rate=0.0,
                total_token_cost_savings=0,
                avg_local_latency_ms=0.0,
                avg_remote_latency_ms=0.0,
                verification_success_rate=0.0,
                routing_breakdown={},
            )

        cache_hits = sum(1 for r in records if r.cache_status == "HIT")
        cache_misses = total - cache_hits
        non_cache_records = [r for r in records if r.selected_route != "CACHE"]

        local_count = sum(1 for r in records if r.selected_route == "LOCAL")
        remote_count = sum(1 for r in records if r.selected_route == "REMOTE")
        non_cache_total = len(non_cache_records)

        local_offload_rate = local_count / non_cache_total if non_cache_total > 0 else 0.0
        remote_fallback_rate = remote_count / non_cache_total if non_cache_total > 0 else 0.0

        total_savings = sum(r.saved_tokens for r in records)

        local_latencies = [r.latency_ms for r in records if r.selected_route == "LOCAL"]
        remote_latencies = [r.latency_ms for r in records if r.selected_route == "REMOTE"]
        avg_local = sum(local_latencies) / len(local_latencies) if local_latencies else 0.0
        avg_remote = sum(remote_latencies) / len(remote_latencies) if remote_latencies else 0.0

        verified = [r for r in records if r.verification_passed is not None]
        verified_ok = sum(1 for r in verified if r.verification_passed)
        verification_success = verified_ok / len(verified) if verified else 0.0

        breakdown: dict[str, int] = {}
        for r in records:
            breakdown[r.task_type] = breakdown.get(r.task_type, 0) + 1

        return DashboardSnapshot(
            cache_hit_ratio=cache_hits / total if total > 0 else 0.0,
            local_offload_rate=local_offload_rate,
            remote_fallback_rate=remote_fallback_rate,
            total_token_cost_savings=total_savings,
            avg_local_latency_ms=avg_local,
            avg_remote_latency_ms=avg_remote,
            verification_success_rate=verification_success,
            routing_breakdown=breakdown,
        )
