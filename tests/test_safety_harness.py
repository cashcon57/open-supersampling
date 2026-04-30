"""Tests for SafetyHarness — must guarantee termination on every exit path.

These tests use a mock LambdaClient (no network) and inject a fake clock so
we can simulate idle / budget / duration scenarios deterministically.
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ors.cloud.lambda_client import LambdaInstance
from ors.cloud.safety_harness import (
    BudgetExceeded,
    HarnessConfig,
    IdleTimeout,
    MaxDurationExceeded,
    SafetyHarness,
)


def _mk_client(active_status: str = "active", ip: str = "10.0.0.1"):
    """Return a MagicMock LambdaClient that simulates a launch+terminate cycle."""
    client = MagicMock()
    client._api_key = "test-key"
    client.launch.return_value = ["i-test-1234"]
    client.terminate.return_value = {"data": {"terminated_instances": [{"id": "i-test-1234"}]}}
    inst = LambdaInstance(
        instance_id="i-test-1234",
        instance_type="gpu_1x_a100",
        region="us-west-1",
        status=active_status,
        ip=ip,
        hostname=None,
        launched_at=None,
    )
    client.get_instance.return_value = inst
    client.list_instance_types.return_value = {}
    return client


def _mk_config(**kwargs) -> HarnessConfig:
    base = dict(
        instance_type="gpu_1x_a100",
        region="us-west-1",
        ssh_key_names=["test-key"],
        max_duration_s=10,
        budget_usd=0.05,  # ~140 sec at $1.29/hr
        idle_timeout_s=10,
        idle_check_interval_s=2,
        watchdog_stale_s=999,  # disable watchdog firing in tests
    )
    base.update(kwargs)
    return HarnessConfig(**base)


def test_harness_terminates_on_normal_exit():
    client = _mk_client()
    cfg = _mk_config()
    with SafetyHarness(client, cfg) as inst:
        assert inst.instance_id == "i-test-1234"
    # On exit, terminate must have been called with the instance ID
    client.terminate.assert_called_with(["i-test-1234"])


def test_harness_terminates_on_exception():
    client = _mk_client()
    cfg = _mk_config()
    try:
        with SafetyHarness(client, cfg):
            raise RuntimeError("simulated training crash")
    except RuntimeError:
        pass
    # Even with exception in the block, terminate must fire
    client.terminate.assert_called_with(["i-test-1234"])


def test_harness_rejects_high_budget_without_override():
    client = _mk_client()
    cfg = _mk_config(budget_usd=100.0)
    with pytest.raises(ValueError, match="exceeds .50 default cap"):
        SafetyHarness(client, cfg)


def test_harness_allows_high_budget_with_explicit_override():
    client = _mk_client()
    cfg = _mk_config(budget_usd=100.0, require_explicit_high_budget=False)
    h = SafetyHarness(client, cfg)
    assert h._config.budget_usd == 100.0


def test_harness_rejects_max_duration_over_24h():
    client = _mk_client()
    cfg = _mk_config(max_duration_s=25 * 3600)
    with pytest.raises(ValueError, match="exceeds 24 hours"):
        SafetyHarness(client, cfg)


def test_check_limits_raises_on_budget_exceeded(monkeypatch):
    client = _mk_client()
    cfg = _mk_config(budget_usd=0.01, max_duration_s=3600)  # tiny budget
    h = SafetyHarness(client, cfg)
    h.__enter__()
    try:
        # Fake elapsed time to force budget exceedance
        h._launch_t = time.time() - 100  # 100 sec elapsed at $1.29/hr ≈ $0.036
        with pytest.raises(BudgetExceeded):
            h.check_limits()
    finally:
        h.__exit__(None, None, None)
    client.terminate.assert_called()


def test_check_limits_raises_on_max_duration_exceeded():
    client = _mk_client()
    cfg = _mk_config(max_duration_s=5)
    h = SafetyHarness(client, cfg)
    h.__enter__()
    try:
        h._launch_t = time.time() - 10  # 10 sec elapsed > 5 sec cap
        with pytest.raises(MaxDurationExceeded):
            h.check_limits()
    finally:
        h.__exit__(None, None, None)


def test_terminate_is_idempotent():
    client = _mk_client()
    cfg = _mk_config()
    h = SafetyHarness(client, cfg)
    h.__enter__()
    h._terminate_idempotent("test_first")
    h._terminate_idempotent("test_second")
    h.__exit__(None, None, None)
    # Should only have called terminate once despite multiple invocations
    assert client.terminate.call_count == 1
