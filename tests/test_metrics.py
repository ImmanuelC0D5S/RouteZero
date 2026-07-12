import pytest
import asyncio
import tempfile
import os
from datetime import datetime, timezone
from routezero.config import Settings
from routezero.metrics import MetricsLogger, RequestLogRecord, DashboardSnapshot


def _temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return path


@pytest.mark.asyncio
async def test_metrics_log_and_snapshot():
    db_path = _temp_db()
    try:
        settings = Settings(sqlite_path=db_path)
        logger = MetricsLogger(settings)
        await logger.init_db()
        
        record = RequestLogRecord(
            request_id="test_001",
            cache_status="MISS",
            selected_route="LOCAL",
            task_type="factual",
            saved_tokens=100,
            latency_ms=50.0,
            verification_passed=True,
            timestamp=datetime.now(timezone.utc),
        )
        await logger.log_request(record)
        
        # Give the background worker time to move the record from queue to batch
        await asyncio.sleep(0.1)
        await logger._flush()
        
        snapshot = await logger.get_dashboard_snapshot()
        assert isinstance(snapshot, DashboardSnapshot)
        assert snapshot.local_offload_rate == 1.0
        assert snapshot.remote_fallback_rate == 0.0
        assert snapshot.cache_hit_ratio == 0.0
        assert snapshot.total_token_cost_savings == 100
    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass


@pytest.mark.asyncio
async def test_metrics_empty_snapshot():
    db_path = _temp_db()
    try:
        settings = Settings(sqlite_path=db_path)
        logger = MetricsLogger(settings)
        await logger.init_db()
        
        snapshot = await logger.get_dashboard_snapshot()
        assert snapshot.cache_hit_ratio == 0.0
        assert snapshot.routing_breakdown == {}
    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass


@pytest.mark.asyncio
async def test_metrics_multiple_records():
    db_path = _temp_db()
    try:
        settings = Settings(sqlite_path=db_path)
        logger = MetricsLogger(settings)
        await logger.init_db()
        
        records = [
            RequestLogRecord("r1", "HIT", "CACHE", "factual", 200, 5.0, True, datetime.now(timezone.utc)),
            RequestLogRecord("r2", "MISS", "LOCAL", "summarize", 150, 50.0, True, datetime.now(timezone.utc)),
            RequestLogRecord("r3", "MISS", "REMOTE", "codegen", 0, 500.0, True, datetime.now(timezone.utc)),
        ]
        for r in records:
            await logger.log_request(r)
        
        await asyncio.sleep(0.15)
        await logger._flush()
        
        snapshot = await logger.get_dashboard_snapshot()
        assert snapshot.cache_hit_ratio == 1.0 / 3.0
        assert snapshot.local_offload_rate == 0.5  # 1 local / 2 non-cache
        assert snapshot.remote_fallback_rate == 0.5  # 1 remote / 2 non-cache
        assert "factual" in snapshot.routing_breakdown
        assert "codegen" in snapshot.routing_breakdown
    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass
