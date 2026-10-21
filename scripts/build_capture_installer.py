"""Per-game capture-installer config builder.

Generates the **input files** that an MSI compiler (WiX / candle+light)
will bundle into a per-game ``oss_capture_<game>_v<ver>.msi``. The MSI
compilation step itself lives outside this script (it requires Windows
tooling); we just produce the JSON config + installer manifest that the
Codex-side DLL build pipeline consumes.

Usage:

    python scripts/build_capture_installer.py \
        --game cyberpunk-2077 \
        --game-exe-name Cyberpunk2077.exe \
        --proxy-dll-name dxgi.dll \
        --output dist/oss_capture_cyberpunk_v1.0.0/

This writes:

    <output>/config.json           (bundled into %LOCALAPPDATA%\\oss-capture\\)
    <output>/installer_manifest.json   (consumed by WiX/candle build)

The ``install_token`` is a freshly minted UUID4 — the unique key that
identifies this installer build to the ingest server.

Note: this is the **build-time** config, baked into the MSI. At
install-time the user agrees to the consent dialog described in the
design memo §"One-click installer".
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse, urlunparse
from xml.sax.saxutils import escape


CAPTURE_API_DEFAULT = "https://capture.oss-supersampling.dev"
CAPTURE_MODES = ("trickle", "lite", "regular", "INSANE")
CAPTURE_STORAGE_MODES = ("local", "upload")
INSANE_SUPERSAMPLE_GT_CONSENT = (
    "INSANE mode runs an automatic 256-frame supersample ground-truth pass "
    "when the camera settles for \u22651.5 s. This briefly stutters the game "
    "(~250 ms) and is the source of OSS's beyond-DLSS quality data. By "
    "accepting INSANE mode you accept this trade-off."
)

_GAME_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_DLL_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}\.dll$")
_EXE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}\.exe$")
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.]+)?$")
LOCALAPPDATA_TOKEN = "LOCALAPPDATA_OSS_CAPTURE"
GAME_BIN_TOKEN = "GAME_BIN_X64_DIR"
DXGI_PROXY_DLL_NAME = "dxgi.dll"
NGX_PROXY_DLL_NAME = "nvngx_dlss.dll"
REQUIRED_ARTIFACTS = (
    DXGI_PROXY_DLL_NAME,
    NGX_PROXY_DLL_NAME,
    "oss_capture_uploader.exe",
    "config.json",
    "scheduled_task.xml",
)

SCHEDULED_TASK_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Uploads OSS capture training frames and deletes local pending files after terminal responses.</Description>
    <Author>OpenSuperSampling</Author>
  </RegistrationInfo>
  <Triggers>
    <TimeTrigger>
      <Repetition>
        <Interval>PT10M</Interval>
        <StopAtDurationEnd>false</StopAtDurationEnd>
      </Repetition>
      <StartBoundary>2026-05-04T00:00:00</StartBoundary>
      <Enabled>true</Enabled>
    </TimeTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>true</RunOnlyIfNetworkAvailable>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <ExecutionTimeLimit>PT5M</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>%LOCALAPPDATA%\\oss-capture\\oss_capture_uploader.exe</Command>
      <Arguments>--once --config "%LOCALAPPDATA%\\oss-capture\\config.json"</Arguments>
      <WorkingDirectory>%LOCALAPPDATA%\\oss-capture</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""


def normalize_capture_api_base(value: str) -> str:
    """Return a base capture API URL, accepting either a base or /ingest URL."""
    parsed = urlparse(value)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"capture_api_base must be an absolute URL: {value!r}")
    path = parsed.path.rstrip("/")
    if path.endswith("/ingest"):
        path = path[: -len("/ingest")]
    elif path == "/ingest":
        path = ""
    return urlunparse(parsed._replace(path=path, params="", query="", fragment="")).rstrip("/")


def build_config(
    *,
    game_id: str,
    game_exe_name: str,
    proxy_dll_name: str,
    installer_version: str,
    capture_api_base: str,
    install_token: Optional[str] = None,
    suggested_capture_rate_per_min: float = 3.0,
    capture_mode: str = "lite",
    capture_storage_mode: str = "local",
) -> Dict[str, Any]:
    """Return the install-time ``config.json`` dict.

    Pure function — no I/O, no UUID side-effect unless ``install_token``
    is None. Useful for unit tests and for scripted batch builds.
    """
    if not _GAME_ID_RE.match(game_id):
        raise ValueError(
            "game_id must match [a-z0-9][a-z0-9._-]{0,63} (lowercase, fs-safe)"
        )
    if not _EXE_NAME_RE.match(game_exe_name):
        raise ValueError(f"game_exe_name {game_exe_name!r} not a valid .exe name")
    if not _DLL_NAME_RE.match(proxy_dll_name):
        raise ValueError(f"proxy_dll_name {proxy_dll_name!r} not a valid .dll name")
    if proxy_dll_name != DXGI_PROXY_DLL_NAME:
        raise ValueError(f"proxy_dll_name must be {DXGI_PROXY_DLL_NAME} for Cyberpunk capture")
    if not _VERSION_RE.match(installer_version):
        raise ValueError(
            f"installer_version {installer_version!r} not semver"
        )
    if capture_mode not in CAPTURE_MODES:
        raise ValueError(f"unknown capture_mode {capture_mode!r}")
    if capture_storage_mode not in CAPTURE_STORAGE_MODES:
        raise ValueError(f"unknown capture_storage_mode {capture_storage_mode!r}")
    if capture_storage_mode == "upload" and install_token is None:
        raise ValueError(
            "upload storage mode requires --install-token from the ingest service"
        )
    capture_api_base = normalize_capture_api_base(capture_api_base)

    if install_token is None:
        install_token = uuid.uuid4().hex

    return {
        "schema_version": 1,
        "game_id": game_id,
        "game_exe_name": game_exe_name,
        "proxy_dll_name": proxy_dll_name,
        "ngx_proxy_dll_name": NGX_PROXY_DLL_NAME,
        "install_token": install_token,
        "installer_version": installer_version,
        "capture_mode": capture_mode,
        "capture_storage_mode": capture_storage_mode,
        "capture_api_base": capture_api_base,
        "ingest_url": capture_api_base + "/ingest",
        "endpoints": {
            "ingest": capture_api_base + "/ingest",
            "session_start": capture_api_base + "/session/start",
            "stats": capture_api_base + "/stats",
        },
        "suggested_capture_rate_per_min": float(suggested_capture_rate_per_min),
        # The DLL+uploader pair reads these knobs at runtime (from
        # %LOCALAPPDATA%\oss-capture\config.json).
        "pending_dir_cap_bytes": 2 * 1024 * 1024 * 1024,  # 2 GB hard cap
        "max_frame_bytes": 16 * 1024 * 1024,  # 16 MB hard cap (server-side too)
        "uploader_retry_attempts": 5,
        "uploader_retry_max_seconds": 30 * 60,  # 30 minutes total
        "consent": {
            "mode": capture_mode,
            "storage_mode": capture_storage_mode,
            "standard_disclosure": (
                (
                    "By installing in upload mode, you agree to upload anonymized "
                    f"gameplay frames from {game_id} to OSS for AI training. "
                    "Captures are deleted from your machine after successful "
                    "upload or terminal rejection."
                )
                if capture_storage_mode == "upload"
                else (
                    f"Local mode stores captures from {game_id} on this machine "
                    "only. The uploader leaves local captures on disk."
                )
            ),
            "insane_supersample_gt_disclosure": (
                INSANE_SUPERSAMPLE_GT_CONSENT if capture_mode == "INSANE" else ""
            ),
        },
    }


def build_installer_manifest(
    config: Dict[str, Any],
    *,
    output_dir: str,
    artifact_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the manifest the WiX build step consumes.

    Lists the files the MSI must bundle, their target paths, and the
    metadata an uninstaller will need (registry keys, scheduled-task
    name, backup behavior for an existing ``dxgi.dll``).
    """
    game_id = config["game_id"]
    artifact_root = artifact_dir or output_dir
    upgrade_code = str(uuid.uuid5(uuid.NAMESPACE_URL, f"oss-capture-installer:{game_id}"))
    return {
        "schema_version": 1,
        "build_dir": output_dir,
        "artifact_dir": artifact_root,
        "config_path": os.path.join(output_dir, "config.json"),
        "scheduled_task_xml_path": os.path.join(output_dir, "scheduled_task.xml"),
        "files": [
            {
                "src": config["proxy_dll_name"],
                "dst_token": GAME_BIN_TOKEN,
                "dst_name": config["proxy_dll_name"],
                "behavior": "backup-existing-then-replace",
            },
            {
                "src": config.get("ngx_proxy_dll_name", NGX_PROXY_DLL_NAME),
                "dst_token": GAME_BIN_TOKEN,
                "dst_name": config.get("ngx_proxy_dll_name", NGX_PROXY_DLL_NAME),
                "behavior": "backup-existing-then-replace",
            },
            {
                "src": "oss_capture_uploader.exe",
                "dst_token": LOCALAPPDATA_TOKEN,
                "dst_name": "oss_capture_uploader.exe",
            },
            {
                "src": "config.json",
                "dst_token": LOCALAPPDATA_TOKEN,
                "dst_name": "config.json",
            },
            {
                "src": "scheduled_task.xml",
                "dst_token": LOCALAPPDATA_TOKEN,
                "dst_name": "scheduled_task.xml",
            },
        ],
        "artifacts": {
            "required": list(REQUIRED_ARTIFACTS),
            "producers": {
                DXGI_PROXY_DLL_NAME: (
                    "cmake --install oss/gaussian/interception/build "
                    "--config Release --component CaptureInstaller --prefix <artifact_dir>; "
                    "copy/rename oss_capture.dll to dxgi.dll"
                ),
                NGX_PROXY_DLL_NAME: (
                    "cmake --install oss/gaussian/interception/build "
                    "--config Release --component CaptureInstaller --prefix <artifact_dir>; "
                    "copy/rename oss_capture_ngx.dll to nvngx_dlss.dll"
                ),
                "oss_capture_uploader.exe": (
                    "pyinstaller oss/capture/installer/oss_capture_uploader_pyinstaller.spec"
                ),
                "config.json": "scripts/build_capture_installer.py writes this file",
                "scheduled_task.xml": "scripts/build_capture_installer.py writes this file",
            },
        },
        "scheduled_task": {
            "name": f"OSS-Capture-Uploader-{game_id}",
            "exe": "oss_capture_uploader.exe",
            "interval_minutes": 10,
            "xml": "scheduled_task.xml",
        },
        "consent": config["consent"],
        "verify": {
            "expected_game_exe": config["game_exe_name"],
            "expected_relative_path": "bin/x64/",
        },
        "uninstall": {
            "restore_backup": True,
            "remove_pending_dir": True,
            "remove_scheduled_task": True,
        },
        "wix": {
            "toolset": "WiX Toolset v3",
            "source_path": os.path.join(output_dir, "oss_capture_installer.wxs"),
            "product_name": f"OSS Capture - {game_id}",
            "manufacturer": "OpenSuperSampling",
            "product_version": config["installer_version"],
            "upgrade_code": upgrade_code,
            "required_properties": {
                GAME_BIN_TOKEN: (
                    "absolute Cyberpunk bin\\x64 directory containing "
                    f"{config['game_exe_name']}"
                )
            },
            "extensions": ["WixUtilExtension"],
        },
    }


def validate_installer_manifest(
    manifest: Dict[str, Any],
    *,
    require_artifacts: bool = False,
    artifact_dir: Optional[str] = None,
) -> List[str]:
    """Return a list of manifest problems; empty means package-ready."""
    errors: List[str] = []
    if manifest.get("schema_version") != 1:
        errors.append("schema_version must be 1")

    files = manifest.get("files")
    if not isinstance(files, list):
        errors.append("files must be a list")
        files = []

    by_src = {f.get("src"): f for f in files if isinstance(f, dict)}
    for required in REQUIRED_ARTIFACTS:
        if required not in by_src:
            errors.append(f"missing required file entry: {required}")

    dxgi_dll = by_src.get(DXGI_PROXY_DLL_NAME, {})
    if dxgi_dll.get("dst_token") != GAME_BIN_TOKEN:
        errors.append(f"{DXGI_PROXY_DLL_NAME} must target {GAME_BIN_TOKEN}")
    if dxgi_dll.get("dst_name") != DXGI_PROXY_DLL_NAME:
        errors.append(f"{DXGI_PROXY_DLL_NAME} dst_name must be {DXGI_PROXY_DLL_NAME}")
    if dxgi_dll.get("behavior") != "backup-existing-then-replace":
        errors.append(f"{DXGI_PROXY_DLL_NAME} must declare backup-existing-then-replace behavior")

    ngx_dll = by_src.get(NGX_PROXY_DLL_NAME, {})
    if ngx_dll.get("dst_token") != GAME_BIN_TOKEN:
        errors.append(f"{NGX_PROXY_DLL_NAME} must target {GAME_BIN_TOKEN}")
    if ngx_dll.get("dst_name") != NGX_PROXY_DLL_NAME:
        errors.append(f"{NGX_PROXY_DLL_NAME} dst_name must be {NGX_PROXY_DLL_NAME}")
    if ngx_dll.get("behavior") != "backup-existing-then-replace":
        errors.append(f"{NGX_PROXY_DLL_NAME} must declare backup-existing-then-replace behavior")

    for src in ("oss_capture_uploader.exe", "config.json", "scheduled_task.xml"):
        entry = by_src.get(src, {})
        if entry.get("dst_token") != LOCALAPPDATA_TOKEN:
            errors.append(f"{src} must target {LOCALAPPDATA_TOKEN}")
        if entry.get("dst_name") != src:
            errors.append(f"{src} dst_name must be {src}")

    task = manifest.get("scheduled_task", {})
    if task.get("exe") != "oss_capture_uploader.exe":
        errors.append("scheduled_task.exe must be oss_capture_uploader.exe")
    if task.get("interval_minutes") != 10:
        errors.append("scheduled_task.interval_minutes must be 10")
    if task.get("xml") != "scheduled_task.xml":
        errors.append("scheduled_task.xml must be scheduled_task.xml")

    verify = manifest.get("verify", {})
    expected_game_exe = verify.get("expected_game_exe")
    if not isinstance(expected_game_exe, str) or not _EXE_NAME_RE.match(expected_game_exe):
        errors.append("verify.expected_game_exe must be a valid .exe filename")
    if verify.get("expected_relative_path") != "bin/x64/":
        errors.append("verify.expected_relative_path must be bin/x64/")

    wix = manifest.get("wix", {})
    if wix.get("toolset") != "WiX Toolset v3":
        errors.append("wix.toolset must be WiX Toolset v3")
    if GAME_BIN_TOKEN not in wix.get("required_properties", {}):
        errors.append(f"wix.required_properties must document {GAME_BIN_TOKEN}")

    if require_artifacts:
        root = Path(artifact_dir or manifest.get("artifact_dir") or manifest.get("build_dir") or ".")
        for required in REQUIRED_ARTIFACTS:
            if not (root / required).is_file():
                errors.append(f"missing artifact: {root / required}")

    return errors


def render_wix_source(manifest: Dict[str, Any]) -> str:
    """Render a WiX v3 source file for the validated manifest."""
    errors = validate_installer_manifest(manifest)
    if errors:
        raise ValueError("invalid installer manifest: " + "; ".join(errors))

    wix = manifest["wix"]
    product_name = escape(wix["product_name"])
    manufacturer = escape(wix["manufacturer"])
    product_version = escape(wix["product_version"])
    upgrade_code = escape(wix["upgrade_code"])
    task_name = escape(manifest["scheduled_task"]["name"])
    proxy_name = escape(next(f["dst_name"] for f in manifest["files"] if f["src"] == DXGI_PROXY_DLL_NAME))
    ngx_proxy_name = escape(next(f["dst_name"] for f in manifest["files"] if f["src"] == NGX_PROXY_DLL_NAME))
    backup_name = escape(f"{proxy_name}.oss-backup")
    ngx_backup_name = escape(f"{ngx_proxy_name}.oss-backup")

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Wix xmlns="http://schemas.microsoft.com/wix/2006/wi">
  <Product Id="*" Name="{product_name}" Language="1033" Version="{product_version}" Manufacturer="{manufacturer}" UpgradeCode="{upgrade_code}">
    <Package InstallerVersion="500" Compressed="yes" InstallScope="perUser" />
    <MajorUpgrade DowngradeErrorMessage="A newer OSS Capture installer is already installed." />
    <MediaTemplate EmbedCab="yes" />

    <Property Id="TASK_NAME" Value="{task_name}" />

    <Directory Id="TARGETDIR" Name="SourceDir">
      <Directory Id="{GAME_BIN_TOKEN}" Name="GameBinX64">
        <Component Id="CaptureProxyDll" Guid="*">
          <File Id="CaptureProxyDllFile" Source="$(var.ArtifactDir)\\{proxy_name}" Name="{proxy_name}" KeyPath="yes" />
        </Component>
        <Component Id="CaptureNgxProxyDll" Guid="*">
          <File Id="CaptureNgxProxyDllFile" Source="$(var.ArtifactDir)\\{ngx_proxy_name}" Name="{ngx_proxy_name}" KeyPath="yes" />
        </Component>
      </Directory>
      <Directory Id="LocalAppDataFolder">
        <Directory Id="OssCaptureDir" Name="oss-capture">
          <Component Id="CaptureUploader" Guid="*">
            <File Id="CaptureUploaderFile" Source="$(var.ArtifactDir)\\oss_capture_uploader.exe" Name="oss_capture_uploader.exe" KeyPath="yes" />
          </Component>
          <Component Id="CaptureConfig" Guid="*">
            <File Id="CaptureConfigFile" Source="$(var.ArtifactDir)\\config.json" Name="config.json" KeyPath="yes" />
          </Component>
          <Component Id="CaptureScheduledTaskXml" Guid="*">
            <File Id="ScheduledTaskXmlFile" Source="$(var.ArtifactDir)\\scheduled_task.xml" Name="scheduled_task.xml" KeyPath="yes" />
          </Component>
        </Directory>
      </Directory>
    </Directory>

    <Feature Id="DefaultFeature" Title="OSS Capture" Level="1">
      <ComponentRef Id="CaptureProxyDll" />
      <ComponentRef Id="CaptureNgxProxyDll" />
      <ComponentRef Id="CaptureUploader" />
      <ComponentRef Id="CaptureConfig" />
      <ComponentRef Id="CaptureScheduledTaskXml" />
    </Feature>

    <CustomAction Id="BackupExistingProxyDll" Directory="{GAME_BIN_TOKEN}" Execute="deferred" Impersonate="yes" Return="ignore"
                  ExeCommand="[SystemFolder]cmd.exe /C if exist &quot;[{GAME_BIN_TOKEN}]{proxy_name}&quot; if not exist &quot;[{GAME_BIN_TOKEN}]{backup_name}&quot; copy /Y &quot;[{GAME_BIN_TOKEN}]{proxy_name}&quot; &quot;[{GAME_BIN_TOKEN}]{backup_name}&quot;" />
    <CustomAction Id="BackupExistingNgxProxyDll" Directory="{GAME_BIN_TOKEN}" Execute="deferred" Impersonate="yes" Return="ignore"
                  ExeCommand="[SystemFolder]cmd.exe /C if exist &quot;[{GAME_BIN_TOKEN}]{ngx_proxy_name}&quot; if not exist &quot;[{GAME_BIN_TOKEN}]{ngx_backup_name}&quot; copy /Y &quot;[{GAME_BIN_TOKEN}]{ngx_proxy_name}&quot; &quot;[{GAME_BIN_TOKEN}]{ngx_backup_name}&quot;" />
    <CustomAction Id="RestoreExistingProxyDll" Directory="{GAME_BIN_TOKEN}" Execute="deferred" Impersonate="yes" Return="ignore"
                  ExeCommand="[SystemFolder]cmd.exe /C if exist &quot;[{GAME_BIN_TOKEN}]{backup_name}&quot; copy /Y &quot;[{GAME_BIN_TOKEN}]{backup_name}&quot; &quot;[{GAME_BIN_TOKEN}]{proxy_name}&quot; &amp; del /F /Q &quot;[{GAME_BIN_TOKEN}]{backup_name}&quot;" />
    <CustomAction Id="RestoreExistingNgxProxyDll" Directory="{GAME_BIN_TOKEN}" Execute="deferred" Impersonate="yes" Return="ignore"
                  ExeCommand="[SystemFolder]cmd.exe /C if exist &quot;[{GAME_BIN_TOKEN}]{ngx_backup_name}&quot; copy /Y &quot;[{GAME_BIN_TOKEN}]{ngx_backup_name}&quot; &quot;[{GAME_BIN_TOKEN}]{ngx_proxy_name}&quot; &amp; del /F /Q &quot;[{GAME_BIN_TOKEN}]{ngx_backup_name}&quot;" />
    <CustomAction Id="RegisterUploaderTask" Directory="OssCaptureDir" Execute="deferred" Impersonate="yes" Return="check"
                  ExeCommand="[SystemFolder]schtasks.exe /Create /TN &quot;[TASK_NAME]&quot; /XML &quot;[#ScheduledTaskXmlFile]&quot; /F" />
    <CustomAction Id="RemoveUploaderTask" Directory="OssCaptureDir" Execute="deferred" Impersonate="yes" Return="ignore"
                  ExeCommand="[SystemFolder]schtasks.exe /Delete /TN &quot;[TASK_NAME]&quot; /F" />
    <InstallExecuteSequence>
      <Custom Action="BackupExistingProxyDll" Before="InstallFiles">NOT Installed</Custom>
      <Custom Action="BackupExistingNgxProxyDll" After="BackupExistingProxyDll">NOT Installed</Custom>
      <Custom Action="RegisterUploaderTask" After="InstallFiles">NOT Installed</Custom>
      <Custom Action="RemoveUploaderTask" Before="RemoveFiles">REMOVE="ALL"</Custom>
      <Custom Action="RestoreExistingProxyDll" After="RemoveFiles">REMOVE="ALL"</Custom>
      <Custom Action="RestoreExistingNgxProxyDll" After="RestoreExistingProxyDll">REMOVE="ALL"</Custom>
    </InstallExecuteSequence>
  </Product>
</Wix>
"""


def find_wix_tools() -> Dict[str, str]:
    """Return paths for WiX v3 + msiexec tools available on this host."""
    tools = {
        "candle": shutil.which("candle.exe") or shutil.which("candle"),
        "light": shutil.which("light.exe") or shutil.which("light"),
        "msiexec": shutil.which("msiexec.exe") or shutil.which("msiexec"),
    }
    return {name: path for name, path in tools.items() if path}


def _copy_or_stub_smoke_artifact(src_root: Path, smoke_root: Path, name: str, payload: bytes) -> None:
    source = src_root / name
    target = smoke_root / name
    if source.is_file():
        shutil.copy2(source, target)
    else:
        target.write_bytes(payload)


def _run_checked(cmd: Sequence[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd),
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def run_fake_cyberpunk_installer_smoke(
    manifest: Dict[str, Any],
    *,
    work_dir: str | Path,
    artifact_dir: str | Path | None = None,
    keep_workdir: bool = False,
) -> Dict[str, Any]:
    """Build and install/uninstall the MSI against a fake Cyberpunk bin/x64 tree.

    The smoke is gated to Windows hosts with WiX v3 and msiexec available. It
    never points the installer at a real game tree: both the Cyberpunk exe and
    pre-existing proxy DLLs are created under a temporary work directory, then
    install/uninstall is verified for both backup and restore behavior.
    """
    if platform.system().lower() != "windows":
        return {"status": "skipped", "reason": "installer smoke requires Windows"}

    tools = find_wix_tools()
    missing_tools = [name for name in ("candle", "light", "msiexec") if name not in tools]
    if missing_tools:
        return {
            "status": "skipped",
            "reason": "WiX v3/msiexec tools unavailable",
            "missing_tools": missing_tools,
        }

    validation_errors = validate_installer_manifest(manifest)
    if validation_errors:
        return {"status": "error", "errors": validation_errors}

    root = Path(work_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="oss-capture-msi-smoke-", dir=str(root)))
    fake_bin = scratch / "Cyberpunk 2077" / "bin" / "x64"
    smoke_artifacts = scratch / "artifacts"
    smoke_artifacts.mkdir(parents=True, exist_ok=True)
    fake_bin.mkdir(parents=True, exist_ok=True)

    expected_game_exe = manifest.get("verify", {}).get("expected_game_exe", "Cyberpunk2077.exe")
    (fake_bin / expected_game_exe).write_text("fake cyberpunk exe\n", encoding="utf-8")
    original_dxgi = b"original dxgi proxy\n"
    original_ngx = b"original nvngx proxy\n"
    (fake_bin / DXGI_PROXY_DLL_NAME).write_bytes(original_dxgi)
    (fake_bin / NGX_PROXY_DLL_NAME).write_bytes(original_ngx)

    source_artifacts = Path(artifact_dir or manifest.get("artifact_dir") or manifest.get("build_dir") or ".")
    _copy_or_stub_smoke_artifact(source_artifacts, smoke_artifacts, DXGI_PROXY_DLL_NAME, b"smoke dxgi artifact\n")
    _copy_or_stub_smoke_artifact(source_artifacts, smoke_artifacts, NGX_PROXY_DLL_NAME, b"smoke nvngx artifact\n")
    _copy_or_stub_smoke_artifact(source_artifacts, smoke_artifacts, "oss_capture_uploader.exe", b"smoke uploader artifact\n")
    for name in ("config.json", "scheduled_task.xml"):
        source = source_artifacts / name
        if source.is_file():
            shutil.copy2(source, smoke_artifacts / name)
        else:
            manifest_key = "config_path" if name == "config.json" else "scheduled_task_xml_path"
            manifest_source = Path(manifest.get(manifest_key, ""))
            if manifest_source.is_file():
                shutil.copy2(manifest_source, smoke_artifacts / name)
            elif name == "config.json":
                (smoke_artifacts / name).write_text("{}", encoding="utf-8")
            else:
                (smoke_artifacts / name).write_text(SCHEDULED_TASK_XML, encoding="utf-8")

    wxs_path = Path(manifest["wix"]["source_path"])
    wixobj_path = scratch / "oss_capture_installer.wixobj"
    msi_path = scratch / "oss_capture_installer.msi"
    command_log: list[dict[str, Any]] = []
    try:
        commands = [
            [
                tools["candle"],
                f"-dArtifactDir={smoke_artifacts}",
                "-out",
                str(wixobj_path),
                str(wxs_path),
            ],
            [
                tools["light"],
                "-ext",
                "WixUtilExtension",
                "-out",
                str(msi_path),
                str(wixobj_path),
            ],
            [
                tools["msiexec"],
                "/i",
                str(msi_path),
                "/qn",
                "/norestart",
                f"{GAME_BIN_TOKEN}={fake_bin}\\",
            ],
        ]
        for cmd in commands:
            proc = _run_checked(cmd, cwd=scratch)
            command_log.append(
                {
                    "cmd": [Path(cmd[0]).name, *cmd[1:]],
                    "returncode": proc.returncode,
                    "stdout": proc.stdout.strip(),
                    "stderr": proc.stderr.strip(),
                }
            )
            if proc.returncode != 0:
                return {"status": "error", "work_dir": str(scratch), "commands": command_log}

        install_errors: list[str] = []
        if (fake_bin / DXGI_PROXY_DLL_NAME).read_bytes() == original_dxgi:
            install_errors.append("dxgi.dll was not replaced during install")
        if (fake_bin / NGX_PROXY_DLL_NAME).read_bytes() == original_ngx:
            install_errors.append("nvngx_dlss.dll was not replaced during install")
        if not (fake_bin / f"{DXGI_PROXY_DLL_NAME}.oss-backup").is_file():
            install_errors.append("dxgi.dll backup was not created")
        if not (fake_bin / f"{NGX_PROXY_DLL_NAME}.oss-backup").is_file():
            install_errors.append("nvngx_dlss.dll backup was not created")
        if install_errors:
            return {
                "status": "error",
                "phase": "install-verify",
                "errors": install_errors,
                "work_dir": str(scratch),
                "commands": command_log,
            }

        uninstall = [
            tools["msiexec"],
            "/x",
            str(msi_path),
            "/qn",
            "/norestart",
            f"{GAME_BIN_TOKEN}={fake_bin}\\",
        ]
        proc = _run_checked(uninstall, cwd=scratch)
        command_log.append(
            {
                "cmd": [Path(uninstall[0]).name, *uninstall[1:]],
                "returncode": proc.returncode,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
            }
        )
        if proc.returncode != 0:
            return {"status": "error", "work_dir": str(scratch), "commands": command_log}

        rollback_errors: list[str] = []
        if (fake_bin / DXGI_PROXY_DLL_NAME).read_bytes() != original_dxgi:
            rollback_errors.append("dxgi.dll was not restored during uninstall")
        if (fake_bin / NGX_PROXY_DLL_NAME).read_bytes() != original_ngx:
            rollback_errors.append("nvngx_dlss.dll was not restored during uninstall")
        if (fake_bin / f"{DXGI_PROXY_DLL_NAME}.oss-backup").exists():
            rollback_errors.append("dxgi.dll backup was not removed during uninstall")
        if (fake_bin / f"{NGX_PROXY_DLL_NAME}.oss-backup").exists():
            rollback_errors.append("nvngx_dlss.dll backup was not removed during uninstall")
        if rollback_errors:
            return {
                "status": "error",
                "phase": "rollback-verify",
                "errors": rollback_errors,
                "work_dir": str(scratch),
                "commands": command_log,
            }

        return {
            "status": "ready",
            "work_dir": str(scratch),
            "fake_game_bin": str(fake_bin),
            "msi_path": str(msi_path),
            "commands": command_log,
        }
    finally:
        if not keep_workdir:
            shutil.rmtree(scratch, ignore_errors=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build_capture_installer",
        description=(
            "Generate per-game capture-installer config + manifest. "
            "MSI compilation is a separate step (WiX/candle)."
        ),
    )
    parser.add_argument("--game", required=True, dest="game_id")
    parser.add_argument(
        "--game-exe-name",
        required=True,
        help="e.g., Cyberpunk2077.exe",
    )
    parser.add_argument(
        "--proxy-dll-name",
        default="dxgi.dll",
        help="Filename to drop our DLL as (default: dxgi.dll).",
    )
    parser.add_argument(
        "--installer-version",
        default="1.0.0",
        help="Semver string baked into the MSI metadata.",
    )
    parser.add_argument(
        "--capture-api-base",
        default=CAPTURE_API_DEFAULT,
    )
    parser.add_argument(
        "--mode",
        choices=list(CAPTURE_MODES),
        default="lite",
        help=(
            "Capture mode preset. Default 'lite' is the 99%% case "
            "(~500 MB/h, v5-temporal-optimized). 'trickle' (~100 MB/h) "
            "for users who don't want to notice it. 'regular' (~2 GB/h) "
            "for material-aware contributors with uncapped fiber. "
            "'INSANE' (~20-50 GB/h) for data-warriors with high-end GPUs "
            "+ uncapped uplink (note: periodic supersample-GT pass briefly "
            "stutters the game when camera is settled)."
        ),
    )
    parser.add_argument(
        "--storage-mode",
        choices=list(CAPTURE_STORAGE_MODES),
        default="local",
        help=(
            "'local' captures to disk and disables upload. 'upload' enables "
            "the scheduled uploader to send accepted captures to ingest/R2."
        ),
    )
    parser.add_argument(
        "--install-token",
        default=None,
        help="Pin this exact token instead of generating a UUID4 "
        "(useful for repeatable test builds).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory. Will be created if missing.",
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help=(
            "Directory that will contain dxgi.dll, nvngx_dlss.dll, and "
            "oss_capture_uploader.exe for WiX. Defaults to --output."
        ),
    )
    parser.add_argument(
        "--require-artifacts",
        action="store_true",
        help=(
            "Fail if the WiX input artifacts are not present in --artifact-dir. "
            "Use this in the final Windows packaging step."
        ),
    )
    parser.add_argument(
        "--smoke-fake-cyberpunk-install",
        action="store_true",
        help=(
            "When running on Windows with WiX v3 and msiexec available, build "
            "the MSI and install/uninstall it against a temporary fake "
            "Cyberpunk bin\\x64 tree to verify DLL backup/rollback."
        ),
    )
    parser.add_argument(
        "--keep-smoke-workdir",
        action="store_true",
        help="Keep the temporary fake-Cyberpunk installer smoke directory.",
    )

    args = parser.parse_args(argv)

    try:
        config = build_config(
            game_id=args.game_id,
            game_exe_name=args.game_exe_name,
            proxy_dll_name=args.proxy_dll_name,
            installer_version=args.installer_version,
            capture_api_base=args.capture_api_base,
            install_token=args.install_token,
            capture_mode=args.mode,
            capture_storage_mode=args.storage_mode,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    os.makedirs(args.output, exist_ok=True)
    artifact_root = args.artifact_dir or args.output
    os.makedirs(artifact_root, exist_ok=True)
    config_path = os.path.join(args.output, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")
    artifact_config_path = os.path.join(artifact_root, "config.json")
    if os.path.abspath(artifact_config_path) != os.path.abspath(config_path):
        with open(artifact_config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)
            f.write("\n")

    scheduled_task_path = os.path.join(args.output, "scheduled_task.xml")
    with open(scheduled_task_path, "w", encoding="utf-8") as f:
        f.write(SCHEDULED_TASK_XML)
    artifact_task_path = os.path.join(artifact_root, "scheduled_task.xml")
    if os.path.abspath(artifact_task_path) != os.path.abspath(scheduled_task_path):
        with open(artifact_task_path, "w", encoding="utf-8") as f:
            f.write(SCHEDULED_TASK_XML)

    manifest = build_installer_manifest(
        config,
        output_dir=args.output,
        artifact_dir=args.artifact_dir,
    )
    validation_errors = validate_installer_manifest(
        manifest,
        require_artifacts=args.require_artifacts,
        artifact_dir=args.artifact_dir,
    )
    if validation_errors:
        for err in validation_errors:
            print(f"error: {err}", file=sys.stderr)
        return 2

    manifest_path = os.path.join(args.output, "installer_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    wix_path = manifest["wix"]["source_path"]
    with open(wix_path, "w", encoding="utf-8") as f:
        f.write(render_wix_source(manifest))

    smoke_result: Dict[str, Any] | None = None
    if args.smoke_fake_cyberpunk_install:
        smoke_result = run_fake_cyberpunk_installer_smoke(
            manifest,
            work_dir=args.output,
            artifact_dir=args.artifact_dir,
            keep_workdir=args.keep_smoke_workdir,
        )
        smoke_path = os.path.join(args.output, "installer_smoke_report.json")
        with open(smoke_path, "w", encoding="utf-8") as f:
            json.dump(smoke_result, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"wrote {smoke_path}")
        if smoke_result.get("status") == "error":
            for err in smoke_result.get("errors", []):
                print(f"error: installer smoke: {err}", file=sys.stderr)
            if not smoke_result.get("errors"):
                print("error: installer smoke failed", file=sys.stderr)
            return 2

    print(f"wrote {config_path}")
    print(f"wrote {scheduled_task_path}")
    print(f"wrote {manifest_path}")
    print(f"wrote {wix_path}")
    print(f"install_token: {config['install_token']}")
    print(
        "next step: validate artifacts, then run candle/light with "
        f"-dArtifactDir={manifest['artifact_dir']} and GAME_BIN_X64_DIR set at install time"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
