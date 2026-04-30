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
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .protocol import CloudClient, CloudInstance


# Audit + heartbeat directories. These were previously named after Lambda
# specifically; the harness is now vendor-agnostic so paths use the generic
# "cloud" namespace. Old `~/.ors-lambda-*` paths are kept readable for any
# tooling that still consumes them, but new writes go to the new locations.
_AUDIT_LOG_PATH = Path.home() / ".ors-cloud-audit.log"
_HEARTBEAT_DIR = Path.home() / ".ors-cloud-heartbeats"


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

    # Human-in-the-loop pre-launch approval. ALWAYS true except in tests or when
    # caller has already shown a cost preview and gotten user confirmation.
    require_pre_launch_approval: bool = True
    purpose: str = ""  # short human-readable description shown in the approval preview


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

    def __init__(self, client: CloudClient, config: HarnessConfig):
        self._client = client
        self._config = config
        self._instance_id: Optional[str] = None
        self._instance_ids: list[str] = []          # ALL launched instance IDs (multi-instance safe)
        self._instance: Optional[CloudInstance] = None
        self._launch_t: Optional[float] = None
        self._idle_streak_s: int = 0
        self._watchdog_proc: Optional[subprocess.Popen] = None
        self._heartbeat_path: Optional[Path] = None
        self._terminated: bool = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_stop = threading.Event()
        self._original_sigint = None
        self._original_sigterm = None

        # Edge case #9: refuse to launch with unknown pricing — budget tracking
        # would silently report $0 and the budget cap would never trigger.
        if client.hourly_rate(config.instance_type) == 0.0:
            raise ValueError(
                f"unknown hourly pricing for instance_type={config.instance_type!r} "
                f"on vendor={client.vendor_name!r}. Add it to the vendor client's "
                f"pricing table before launching, or budget caps will silently fail."
            )

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
        if self._config.require_pre_launch_approval:
            self._warn_existing_instances()
            self._pre_launch_approval()
        self._install_signal_handlers()
        atexit.register(self._terminate_idempotent, "atexit")
        self._launch()
        self._start_heartbeat_thread()
        self._start_watchdog()
        self._install_self_terminate_on_instance()
        return self._instance  # type: ignore[return-value]

    def _warn_existing_instances(self):
        """Edge case #11: warn if other instances are already running on this
        API key — concurrent harnesses don't share a budget cap."""
        try:
            existing = self._client.list_instances()
            active = [i for i in existing if i.status not in ("terminated", "terminating")]
            if active:
                vendor = self._client.vendor_name
                print()
                print(f"  ⚠ WARNING: {len(active)} instance(s) already active on this {vendor} API key:")
                for inst in active:
                    r = self._client.hourly_rate(inst.instance_type)
                    print(f"    - {inst.instance_id}  {inst.instance_type}  status={inst.status}  ${r}/hr")
                print("  Concurrent launches do NOT share budget caps.")
                print(f"  If these aren't yours, run: python -m scripts.{vendor}_terminate_all")
                print()
        except Exception as e:
            sys.stderr.write(f"[SafetyHarness] WARNING: couldn't list existing instances: {e}\n")

    def _pre_launch_approval(self):
        """Print a cost preview and require interactive confirmation."""
        cfg = self._config
        rate = self._client.hourly_rate(cfg.instance_type)
        max_cost_at_duration = (cfg.max_duration_s / 3600.0) * rate

        preview = [
            "",
            "=" * 70,
            f"  {self._client.vendor_name.upper()} GPU LAUNCH — APPROVAL REQUIRED",
            "=" * 70,
            f"  Purpose         : {cfg.purpose or '(not specified)'}",
            f"  Instance type   : {cfg.instance_type}",
            f"  Region          : {cfg.region}",
            f"  SSH key(s)      : {', '.join(cfg.ssh_key_names)}",
            "",
            f"  Hourly rate     : ${rate:.2f}/hr",
            f"  Budget cap      : ${cfg.budget_usd:.2f}    (auto-terminate at this $)",
            f"  Max duration    : {cfg.max_duration_s/3600:.1f}h    (auto-terminate at this time)",
            f"  Worst-case cost : ${max_cost_at_duration:.2f}    (max_duration × hourly rate)",
            "",
            f"  Idle detection  : terminate after {cfg.idle_timeout_s/60:.0f}min of <{cfg.idle_threshold_pct}% GPU util",
            f"  Watchdog        : terminate if heartbeat stale >{cfg.watchdog_stale_s}s",
            "",
            "  Multiple termination triggers active:",
            "    1. context-manager exit (always)",
            "    2. SIGINT/SIGTERM signal handlers",
            "    3. atexit handler (interpreter shutdown)",
            "    4. external watchdog process (poll heartbeat)",
            "    5. budget cap (cumulative cost)",
            "    6. max duration cap (wall time)",
            "    7. idle timeout (low GPU util)",
            "",
            f"  Audit log       : {cfg.audit_log_path}",
            "=" * 70,
            "",
        ]
        print("\n".join(preview))

        # Auto-approve only if explicitly env-var-disabled (e.g. inside a CLI that
        # has already shown the preview to the user). Both the legacy Lambda-only
        # flag and the new generic flag are honored, so existing tooling keeps
        # working.
        if os.environ.get("ORS_CLOUD_AUTO_APPROVE") == "1" or os.environ.get("ORS_LAMBDA_AUTO_APPROVE") == "1":
            print("  [ORS_CLOUD_AUTO_APPROVE=1 — proceeding without prompt]")
            return

        try:
            answer = input("  Type 'launch' to confirm, anything else to abort: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            raise RuntimeError("launch aborted (no tty / interrupt during approval)")
        if answer != "launch":
            raise RuntimeError(f"launch aborted by user (typed {answer!r})")
        print("  ✓ approved\n")

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self._terminate_idempotent("context_exit")
        finally:
            self._stop_heartbeat_thread()
            self._stop_watchdog()
            self._restore_signal_handlers()
        return False  # never suppress exceptions

    def heartbeat(self):
        """Advance the watchdog timer. Manual heartbeat from training loop.

        The harness ALSO runs an internal background thread that ticks the
        heartbeat unconditionally — see `_start_heartbeat_thread`. So even if
        the user forgets to call this, the watchdog will still see fresh
        heartbeats while the Python interpreter is alive.
        """
        if self._heartbeat_path is not None:
            self._heartbeat_path.write_text(str(time.time()))

    def _start_heartbeat_thread(self):
        """Edge case #7: a daemon thread ticks the heartbeat file on a fixed
        cadence regardless of caller activity. Without this, a long compute
        step in the main thread (>watchdog_stale_s seconds with no
        `heartbeat()` calls) would cause the watchdog to false-positive and
        kill an actively-training instance.

        The thread exits when `_heartbeat_stop` is set or when the interpreter
        shuts down (daemon=True)."""
        if self._heartbeat_path is None:
            return
        interval = max(1, self._config.heartbeat_interval_s)
        stop_evt = self._heartbeat_stop
        path = self._heartbeat_path

        def _tick():
            while not stop_evt.wait(interval):
                try:
                    path.write_text(str(time.time()))
                except Exception:
                    pass

        self._heartbeat_thread = threading.Thread(
            target=_tick, name="ors-heartbeat", daemon=True
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat_thread(self):
        if self._heartbeat_thread is None:
            return
        self._heartbeat_stop.set()
        try:
            self._heartbeat_thread.join(timeout=2)
        except Exception:
            pass
        self._heartbeat_thread = None

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
        rate = self._client.hourly_rate(cfg.instance_type)
        if rate == 0.0:
            sys.stderr.write(
                f"[SafetyHarness] WARNING: unknown pricing for {cfg.instance_type}; "
                "budget tracking will be inaccurate.\n"
            )
        _audit(
            "pre_launch",
            {
                "vendor": self._client.vendor_name,
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
            raise RuntimeError(
                f"{self._client.vendor_name} launch returned no instance IDs"
            )
        # Edge case #8: track ALL returned IDs in case quantity > 1.
        # Currently we always launch quantity=1 but defensive nonetheless.
        self._instance_ids = list(ids)
        self._instance_id = ids[0]
        self._launch_t = time.time()
        _audit("launched", {"instance_ids": self._instance_ids})

        # Wait for instance to become active. Default 15-minute deadline —
        # Lambda A100/H100 in capacity-tight regions can take 5-12 minutes.
        # Print progress every 30s so the user sees the harness is alive.
        boot_timeout_s = 15 * 60
        deadline = time.time() + boot_timeout_s
        last_status = None
        last_log_t = time.time()
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
                print(
                    f"[SafetyHarness] instance {self._instance_id} active at {inst.ip} "
                    f"after {time.time() - self._launch_t:.0f}s"
                )
                return
            if inst.status in ("terminated", "failed", "unhealthy"):
                self._terminated = True
                raise RuntimeError(f"instance entered status={inst.status} during boot")
            # Periodic progress log so long boots don't appear hung
            now = time.time()
            if inst.status != last_status or (now - last_log_t) >= 30:
                elapsed = now - self._launch_t
                remaining = deadline - now
                print(
                    f"[SafetyHarness] {self._instance_id} status={inst.status} "
                    f"after {elapsed:.0f}s, {remaining:.0f}s remaining before boot timeout",
                    flush=True,
                )
                last_status = inst.status
                last_log_t = now
            time.sleep(10)
        # Timed out waiting for active — terminate to avoid orphaning.
        self._terminate_idempotent("boot_timeout")
        raise RuntimeError(
            f"instance did not reach 'active' within {boot_timeout_s/60:.0f} minutes"
        )

    def _terminate_idempotent(self, reason: str):
        if self._terminated or not self._instance_ids:
            return
        self._terminated = True

        # Edge case #8: terminate ALL tracked instance IDs, not just the first.
        ids_to_kill = list(self._instance_ids)
        try:
            self._client.terminate(ids_to_kill)
            _audit(
                "terminated",
                {
                    "instance_ids": ids_to_kill,
                    "reason": reason,
                    "elapsed_s": self.elapsed_s,
                    "cost_usd": self._cost_so_far(),
                },
            )
        except Exception as e:
            _audit(
                "terminate_failed",
                {"instance_ids": ids_to_kill, "reason": reason, "error": str(e)},
            )
            sys.stderr.write(
                f"[SafetyHarness] CRITICAL: terminate failed for {ids_to_kill}: {e}\n"
                f"  Run `python -m scripts.lambda_terminate_all` to clean up.\n"
            )

        # Edge case #6: verify termination took effect by polling list_instances.
        # If the instance is still in our active list, retry a few times before
        # giving up loudly.
        self._verify_terminated(ids_to_kill)

    def _verify_terminated(self, ids: list[str], retries: int = 5, delay_s: float = 6.0):
        """Confirm the instances no longer appear active. Retry terminate on failure."""
        for attempt in range(retries):
            try:
                active = {i.instance_id for i in self._client.list_instances()
                          if i.status not in ("terminated", "terminating")}
            except Exception as e:
                sys.stderr.write(
                    f"[SafetyHarness] couldn't verify termination (attempt {attempt+1}/{retries}): {e}\n"
                )
                time.sleep(delay_s)
                continue
            still_alive = [i for i in ids if i in active]
            if not still_alive:
                _audit("terminate_verified", {"instance_ids": ids, "attempts": attempt + 1})
                return
            sys.stderr.write(
                f"[SafetyHarness] WARNING: {still_alive} still active after terminate "
                f"(attempt {attempt+1}/{retries}); retrying.\n"
            )
            try:
                self._client.terminate(still_alive)
            except Exception:
                pass
            time.sleep(delay_s)
        vendor = self._client.vendor_name
        sys.stderr.write(
            f"[SafetyHarness] CRITICAL: instance(s) {ids} still appear active after "
            f"{retries} retry attempts. Manually verify with `python -m scripts.{vendor}_status` "
            f"and force-kill with `python -m scripts.{vendor}_terminate_all`.\n"
        )
        _audit("terminate_unverified", {"instance_ids": ids, "retries": retries})

    def _cost_so_far(self) -> float:
        rate = self._client.hourly_rate(self._config.instance_type)
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

    def _install_self_terminate_on_instance(self):
        """Edge cases #4 + #5: ULTIMATE fail-safe. Install a script ON THE
        INSTANCE that calls Lambda's terminate API on itself after
        `max_duration_s` seconds, regardless of what happens on our local
        machine.

        If our laptop catches fire, our network goes down, the watchdog dies,
        and every other layer fails — the instance still kills itself. This
        is the only protection against "local infrastructure totally
        unavailable" scenarios.

        The script is written to /tmp on the instance with mode 0600. The API
        key is passed via SSH stdin (never on the cmdline). The script runs
        with nohup+disown so it survives SSH disconnect.
        """
        if not self._instance or not self._instance.ip or not self._config.ssh_key_path:
            sys.stderr.write(
                "[SafetyHarness] WARNING: self-terminate skipped — no IP or SSH key path.\n"
                "  Local-only protection is active; instance has NO autonomous kill-switch.\n"
            )
            return

        # The script runs entirely on the instance. It sleeps, then POSTs
        # terminate. A buffer is added on top of max_duration_s so we don't
        # race with the harness's own termination at the same wall-clock.
        cap_s = self._config.max_duration_s + 300  # 5-min grace window
        instance_id = self._instance_id
        api_key = self._client._api_key  # noqa: passed via stdin, never on cmdline

        # Vendor-agnostic curl assembly:
        #   - body comes from `client.terminate_request_body(instance_id)`
        #   - either an `Authorization` header or a `-u user:pass` flag is used
        #   - endpoint URL comes from `client.terminate_endpoint()`
        # The actual API key is loaded at firing time from a 0600 file on the
        # instance — never embedded in the script body.
        endpoint = self._client.terminate_endpoint()
        # Build with API_KEY placeholder; the script substitutes from $API_KEY.
        auth_header = self._client.terminate_auth_header("$API_KEY")
        curl_user = self._client.terminate_curl_auth_flag("$API_KEY")
        body = self._client.terminate_request_body(str(instance_id))
        # Escape body for safe embedding in a bash double-quoted string.
        body_bash = body.replace("\\", "\\\\").replace("\"", "\\\"").replace("$", "\\$")

        curl_cmd_parts = ["curl -sS -X POST"]
        if curl_user:
            curl_cmd_parts.append(f'-u "{curl_user}"')
        if auth_header:
            curl_cmd_parts.append(f'-H "{auth_header}"')
        curl_cmd_parts.extend([
            f'"{endpoint}"',
            "-H 'Content-Type: application/json'",
            f'-d "{body_bash}"',
            ">> /tmp/ors-self-terminate.log 2>&1",
        ])
        curl_line = " ".join(curl_cmd_parts)

        remote_script = (
            "#!/usr/bin/env bash\n"
            "set -uo pipefail\n"
            "KEY_FILE=/tmp/.ors-cloud-key\n"
            "INSTANCE_ID=" + str(instance_id) + "\n"
            "MAX_S=" + str(cap_s) + "\n"
            "VENDOR=" + self._client.vendor_name + "\n"
            "echo \"[ors-self-terminate $VENDOR] start $(date -u +%FT%TZ) max_s=$MAX_S\" >> /tmp/ors-self-terminate.log\n"
            "sleep $MAX_S\n"
            "if [ ! -f \"$KEY_FILE\" ]; then\n"
            "  echo \"[ors-self-terminate] missing key file at firing time\" >> /tmp/ors-self-terminate.log\n"
            "  exit 1\n"
            "fi\n"
            "API_KEY=$(cat \"$KEY_FILE\")\n"
            "echo \"[ors-self-terminate $VENDOR] firing at $(date -u +%FT%TZ)\" >> /tmp/ors-self-terminate.log\n"
            + curl_line + "\n"
            "shred -u \"$KEY_FILE\" 2>/dev/null || rm -f \"$KEY_FILE\"\n"
        )

        ssh_base = [
            "ssh",
            "-i", str(self._config.ssh_key_path),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=20",
            "-o", "BatchMode=yes",
            f"{self._config.ssh_user}@{self._instance.ip}",
        ]

        try:
            # Step 1: write the API key to /tmp/.ors-cloud-key with 0600
            r1 = subprocess.run(
                ssh_base + ["bash", "-c", "umask 077; cat > /tmp/.ors-cloud-key && chmod 600 /tmp/.ors-cloud-key"],
                input=api_key,
                text=True,
                capture_output=True,
                timeout=30,
            )
            if r1.returncode != 0:
                raise RuntimeError(f"key upload failed (rc={r1.returncode}): {r1.stderr}")

            # Step 2: write the self-terminate script with 0700
            r2 = subprocess.run(
                ssh_base + ["bash", "-c", "umask 077; cat > /tmp/ors-self-terminate.sh && chmod 700 /tmp/ors-self-terminate.sh"],
                input=remote_script,
                text=True,
                capture_output=True,
                timeout=30,
            )
            if r2.returncode != 0:
                raise RuntimeError(f"script upload failed (rc={r2.returncode}): {r2.stderr}")

            # Step 3: run it under nohup+disown so it survives SSH disconnect
            r3 = subprocess.run(
                ssh_base + [
                    "bash", "-c",
                    "nohup /tmp/ors-self-terminate.sh > /tmp/ors-self-terminate.boot.log 2>&1 & disown; echo $!",
                ],
                capture_output=True, text=True, timeout=20,
            )
            if r3.returncode != 0:
                raise RuntimeError(f"script launch failed (rc={r3.returncode}): {r3.stderr}")

            _audit(
                "self_terminate_installed",
                {
                    "instance_id": self._instance_id,
                    "ip": self._instance.ip,
                    "cap_s": cap_s,
                    "remote_pid": r3.stdout.strip(),
                },
            )
            print(
                f"[SafetyHarness] on-instance self-terminate installed "
                f"(fires at +{cap_s/3600:.1f}h, pid={r3.stdout.strip()})"
            )
        except Exception as e:
            sys.stderr.write(
                f"[SafetyHarness] WARNING: failed to install on-instance self-terminate: {e}\n"
                f"  Local-only protection is active. The watchdog + harness will still\n"
                f"  catch normal failures, but if your local machine becomes unreachable\n"
                f"  before max_duration_s, the instance will keep billing.\n"
            )
            _audit(
                "self_terminate_install_failed",
                {"instance_id": self._instance_id, "error": str(e)},
            )

    def _start_watchdog(self):
        """Spawn an external watchdog process that kills the instance if our
        heartbeat goes stale. This protects against ungraceful main-process death."""
        _HEARTBEAT_DIR.mkdir(parents=True, exist_ok=True)
        self._heartbeat_path = _HEARTBEAT_DIR / f"{self._instance_id}.beat"
        self.heartbeat()  # write initial timestamp

        env = os.environ.copy()
        # Pass API key via env to watchdog (don't expose on cmdline). Both the
        # legacy LAMBDA_API_KEY and the vendor-specific name are set so the
        # watchdog can pick up either depending on `--vendor`.
        vendor = self._client.vendor_name
        if vendor == "lambda":
            env["LAMBDA_API_KEY"] = self._client._api_key  # noqa
        elif vendor == "runpod":
            env["RUNPOD_API_KEY"] = self._client._api_key  # noqa
        else:
            env[f"{vendor.upper()}_API_KEY"] = self._client._api_key  # noqa
        self._watchdog_proc = subprocess.Popen(
            [
                sys.executable, "-m", "ors.cloud.watchdog",
                "--vendor", vendor,
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
