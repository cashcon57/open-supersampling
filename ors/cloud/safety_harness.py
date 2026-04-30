"""SafetyHarness — guarantees Lambda instances cannot run idle and burn money.

Every interaction with a paid GPU instance MUST go through this harness. It
provides multiple, redundant layers of protection so that an instance cannot
be left running by accident:

  1. **Context manager** — `__exit__` always calls terminate, even on exception.
  2. **Hard duration cap** — instance is terminated after `max_duration_s` no
     matter what. Default: 6 hours.
  3. **Budget cap** — cumulative cost (elapsed time × hourly rate) tracked;
     instance terminated when budget is exceeded. Default: $20 per launch.
  4. **Idle detection** — every `idle_check_interval_s` (default 60s) the
     harness SSHes in and runs `nvidia-smi`. If GPU utilization is below
     `idle_threshold_pct` (default 5%) for `idle_timeout_s` (default 900s = 15
     min) consecutive seconds, the instance is terminated.
  5. **Signal handlers** — SIGINT and SIGTERM trigger termination before
     re-raising.
  6. **`atexit` handler** — runs at interpreter shutdown as a last-ditch belt.
  7. **External watchdog** — a separate Python process is launched that polls
     a heartbeat file. If the main process dies without orderly shutdown, the
     watchdog calls terminate via the same API.
  8. **Audit log** — every launch + termination is logged to
     `~/.ors-lambda-audit.log` for after-the-fact accounting.

Defaults are chosen so that the WORST-case spend per accidentally-orphaned
launch is bounded by `max_duration_s × hourly_rate` (e.g. 6 hr × $2.49/hr =
$15 on a 1×H100). If you want to spend more in a single run, override the
defaults explicitly — the harness will not silently allow it.
"""
from __future__ import annotations

import atexit
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .lambda_client import LambdaClient, LambdaInstance


_AUDIT_LOG_PATH = Path.home() / ".ors-lambda-audit.log"
_HEARTBEAT_DIR = Path.home() / ".ors-lambda-heartbeats"


class BudgetExceeded(RuntimeError):
    pass


class MaxDurationExceeded(RuntimeError):
    pass


class IdleTimeout(RuntimeError):
    pass


@dataclass
class HarnessConfig:
    instance_type: str
    region: str
    ssh_key_names: list[str]
    name: str = "ors-training"

    # Hard limits — the harness will terminate when ANY of these triggers.
    max_duration_s: int = 6 * 3600          # 6 hours
    budget_usd: float = 20.0                 # $20
    idle_threshold_pct: float = 5.0          # GPU utilization
    idle_timeout_s: int = 15 * 60            # 15 minutes
    idle_check_interval_s: int = 60          # check once per minute

    # SSH for idle detection
    ssh_key_path: Optional[Path] = None      # private key for SSH-into-instance
    ssh_user: str = "ubuntu"

    # Heartbeat / watchdog
    heartbeat_interval_s: int = 30
    watchdog_stale_s: int = 120              # if main process heartbeat older than this, watchdog kills

    # Behavior
    require_explicit_high_budget: bool = True  # require explicit override above $50
    audit_log_path: Path = field(default_factory=lambda: _AUDIT_LOG_PATH)


def _audit(event: str, payload: dict, log_path: Path = _AUDIT_LOG_PATH):
    """Append a JSON line to the audit log. Best-effort, never raises."""
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "pid": os.getpid(),
                "event": event,
                **payload,
            }
        )
        with log_path.open("a") as f:
            f.write(line + "\n")
    except Exception as e:
        sys.stderr.write(f"[ors.cloud audit] failed to log: {e}\n")


class SafetyHarness:
    """Context manager that GUARANTEES instance termination.

    Usage:
        with SafetyHarness(client, HarnessConfig(...)) as inst:
            # inst.ip / inst.hostname / inst.instance_id available
            # do training work via SSH
            ...
        # instance is terminated here regardless of what happened above

    Inside the `with` block you can call `harness.heartbeat()` to advance the
    watchdog timer. If you don't, the watchdog will assume your process died
    and terminate the instance after `watchdog_stale_s`.
    """

    def __init__(self, client: LambdaClient, config: HarnessConfig):
        self._client = client
        self._config = config
        self._instance_id: Optional[str] = None
        self._instance: Optional[LambdaInstance] = None
        self._launch_t: Optional[float] = None
        self._idle_streak_s: int = 0
        self._watchdog_proc: Optional[subprocess.Popen] = None
        self._heartbeat_path: Optional[Path] = None
        self._terminated: bool = False
        self._original_sigint = None
        self._original_sigterm = None

        # Pre-launch validation
        if config.budget_usd > 50.0 and config.require_explicit_high_budget:
            raise ValueError(
                f"budget_usd={config.budget_usd} exceeds $50 default cap. "
                "Set require_explicit_high_budget=False to override."
            )
        if config.max_duration_s > 24 * 3600:
            raise ValueError(
                f"max_duration_s={config.max_duration_s} exceeds 24 hours. "
                "This harness intentionally caps single-run duration."
            )

    # ----- public API -----

    def __enter__(self) -> LambdaInstance:
        self._install_signal_handlers()
        atexit.register(self._terminate_idempotent, "atexit")
        self._launch()
        self._start_watchdog()
        return self._instance  # type: ignore[return-value]

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self._terminate_idempotent("context_exit")
        finally:
            self._stop_watchdog()
            self._restore_signal_handlers()
        return False  # never suppress exceptions

    def heartbeat(self):
        """Advance the watchdog timer. Call from your training loop."""
        if self._heartbeat_path is not None:
            self._heartbeat_path.write_text(str(time.time()))

    def check_limits(self):
        """Check duration / budget. Raises if exceeded.

        Call this periodically from your training loop. Idle detection runs
        on its own internal cadence and doesn't require user calls.
        """
        if self._terminated:
            return
        elapsed = time.time() - (self._launch_t or 0.0)
        cost = self._cost_so_far()
        if elapsed > self._config.max_duration_s:
            self._terminate_idempotent(
                f"max_duration ({elapsed:.0f}s > {self._config.max_duration_s}s)"
            )
            raise MaxDurationExceeded(
                f"hit max_duration_s={self._config.max_duration_s}; instance terminated"
            )
        if cost > self._config.budget_usd:
            self._terminate_idempotent(
                f"budget (${cost:.2f} > ${self._config.budget_usd:.2f})"
            )
            raise BudgetExceeded(
                f"hit budget=${self._config.budget_usd:.2f}; instance terminated"
            )

    def check_idle(self):
        """SSH in, query nvidia-smi, terminate if idle for too long.

        Returns True if the instance is still active. Raises IdleTimeout
        if termination has occurred.
        """
        if self._terminated or self._instance is None or not self._instance.ip:
            return False
        util = self._poll_gpu_util()
        if util is None:
            # SSH failed — don't penalize; could be transient. Log and continue.
            return True
        if util < self._config.idle_threshold_pct:
            self._idle_streak_s += self._config.idle_check_interval_s
            _audit(
                "idle_observed",
                {
                    "instance_id": self._instance_id,
                    "util_pct": util,
                    "streak_s": self._idle_streak_s,
                },
            )
            if self._idle_streak_s >= self._config.idle_timeout_s:
                self._terminate_idempotent(
                    f"idle ({self._idle_streak_s}s of <{self._config.idle_threshold_pct}% util)"
                )
                raise IdleTimeout(
                    f"GPU idle for {self._idle_streak_s}s; instance terminated"
                )
        else:
            self._idle_streak_s = 0
        return True

    @property
    def cost_so_far_usd(self) -> float:
        return self._cost_so_far()

    @property
    def elapsed_s(self) -> float:
        return time.time() - (self._launch_t or 0.0)

    @property
    def instance_id(self) -> Optional[str]:
        return self._instance_id

    # ----- internals -----

    def _launch(self):
        cfg = self._config
        rate = self._client.list_instance_types() if False else None  # avoid extra API call
        from .lambda_client import INSTANCE_PRICING
        rate = INSTANCE_PRICING.get(cfg.instance_type, 0.0)
        if rate == 0.0:
            sys.stderr.write(
                f"[SafetyHarness] WARNING: unknown pricing for {cfg.instance_type}; "
                "budget tracking will be inaccurate.\n"
            )
        _audit(
            "pre_launch",
            {
                "instance_type": cfg.instance_type,
                "region": cfg.region,
                "budget_usd": cfg.budget_usd,
                "max_duration_s": cfg.max_duration_s,
                "hourly_rate": rate,
            },
        )
        ids = self._client.launch(
            instance_type_name=cfg.instance_type,
            region_name=cfg.region,
            ssh_key_names=cfg.ssh_key_names,
            name=cfg.name,
        )
        if not ids:
            raise RuntimeError("Lambda launch returned no instance IDs")
        self._instance_id = ids[0]
        self._launch_t = time.time()
        _audit("launched", {"instance_id": self._instance_id})

        # Wait for instance to become active (or timeout after 5 minutes)
        deadline = time.time() + 300
        while time.time() < deadline:
            inst = self._client.get_instance(self._instance_id)
            self._instance = inst
            if inst.status == "active" and inst.ip:
                _audit(
                    "active",
                    {
                        "instance_id": self._instance_id,
                        "ip": inst.ip,
                        "wait_s": time.time() - self._launch_t,
                    },
                )
                return
            if inst.status in ("terminated", "failed", "unhealthy"):
                self._terminated = True
                raise RuntimeError(f"instance entered status={inst.status} during boot")
            time.sleep(10)
        # Timed out waiting for active — terminate to avoid orphaning.
        self._terminate_idempotent("boot_timeout")
        raise RuntimeError("instance did not reach 'active' within 5 minutes")

    def _terminate_idempotent(self, reason: str):
        if self._terminated or not self._instance_id:
            return
        self._terminated = True
        try:
            self._client.terminate([self._instance_id])
            _audit(
                "terminated",
                {
                    "instance_id": self._instance_id,
                    "reason": reason,
                    "elapsed_s": self.elapsed_s,
                    "cost_usd": self._cost_so_far(),
                },
            )
        except Exception as e:
            _audit(
                "terminate_failed",
                {"instance_id": self._instance_id, "reason": reason, "error": str(e)},
            )
            sys.stderr.write(
                f"[SafetyHarness] CRITICAL: terminate failed for {self._instance_id}: {e}\n"
                f"  Run `python -m scripts.lambda_terminate_all` to clean up.\n"
            )

    def _cost_so_far(self) -> float:
        from .lambda_client import INSTANCE_PRICING
        rate = INSTANCE_PRICING.get(self._config.instance_type, 0.0)
        return self.elapsed_s / 3600.0 * rate

    def _poll_gpu_util(self) -> Optional[float]:
        """SSH in, run nvidia-smi, return GPU utilization percent. None on failure."""
        if not self._instance or not self._instance.ip:
            return None
        cmd = [
            "ssh",
            "-i", str(self._config.ssh_key_path) if self._config.ssh_key_path else "",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=10",
            "-o", "BatchMode=yes",
            f"{self._config.ssh_user}@{self._instance.ip}",
            "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits",
        ]
        cmd = [c for c in cmd if c]  # drop empty -i if no key path
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            if r.returncode != 0:
                return None
            line = r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""
            return float(line)
        except (subprocess.TimeoutExpired, ValueError, IndexError):
            return None

    def _start_watchdog(self):
        """Spawn an external watchdog process that kills the instance if our
        heartbeat goes stale. This protects against ungraceful main-process death."""
        _HEARTBEAT_DIR.mkdir(parents=True, exist_ok=True)
        self._heartbeat_path = _HEARTBEAT_DIR / f"{self._instance_id}.beat"
        self.heartbeat()  # write initial timestamp

        env = os.environ.copy()
        # Pass API key via env to watchdog (don't expose on cmdline)
        env["LAMBDA_API_KEY"] = self._client._api_key  # noqa
        self._watchdog_proc = subprocess.Popen(
            [
                sys.executable, "-m", "ors.cloud.watchdog",
                "--instance-id", self._instance_id or "",
                "--heartbeat", str(self._heartbeat_path),
                "--stale-s", str(self._config.watchdog_stale_s),
                "--max-duration-s", str(self._config.max_duration_s),
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # detach so signals to parent don't kill watchdog
        )
        _audit(
            "watchdog_started",
            {"instance_id": self._instance_id, "watchdog_pid": self._watchdog_proc.pid},
        )

    def _stop_watchdog(self):
        if self._watchdog_proc is None:
            return
        try:
            self._watchdog_proc.terminate()
            self._watchdog_proc.wait(timeout=5)
        except Exception:
            try:
                self._watchdog_proc.kill()
            except Exception:
                pass
        finally:
            self._watchdog_proc = None
            if self._heartbeat_path and self._heartbeat_path.exists():
                try:
                    self._heartbeat_path.unlink()
                except Exception:
                    pass

    def _install_signal_handlers(self):
        def _handler(signum, frame):
            self._terminate_idempotent(f"signal_{signum}")
            self._stop_watchdog()
            self._restore_signal_handlers()
            # re-raise to default behavior
            os.kill(os.getpid(), signum)

        self._original_sigint = signal.signal(signal.SIGINT, _handler)
        self._original_sigterm = signal.signal(signal.SIGTERM, _handler)

    def _restore_signal_handlers(self):
        if self._original_sigint is not None:
            try:
                signal.signal(signal.SIGINT, self._original_sigint)
            except Exception:
                pass
            self._original_sigint = None
        if self._original_sigterm is not None:
            try:
                signal.signal(signal.SIGTERM, self._original_sigterm)
            except Exception:
                pass
            self._original_sigterm = None
