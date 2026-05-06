"""OSS Capture tray app — system-tray UI for the local-only POC.

The user installs this app on their Windows machine. It runs in the system
tray (carrot menu in the taskbar). When a supported game launches, the
app injects the OSS capture DLL with the user's chosen mode and output
drive baked in. Captures land directly on the local drive — no upload, no
server, no token system. (That mode is for the multi-user shipping
version; this app is the single-machine POC that runs on Cash's 3080 Ti.)

What this file implements (the v0 UI shell):
  - System-tray icon with menu items for status, mode switcher, output-
    drive selection, pause toggle, and quit.
  - Background timer that periodically refreshes drive-space stats and
    runs the disk-cap janitor.
  - Persistent config via ``oss.capture.tray.config``.
  - Steam allowlist menu, process watcher, per-session config write, and
    DLL injection for allowlisted/enabled game launches.

Entry point: ``python -m oss.capture.tray``.
"""
from __future__ import annotations

import logging
import platform
import sys
import threading
from typing import List, Optional

try:
    import pystray  # type: ignore[import-not-found]
    from PIL import Image, ImageDraw  # type: ignore[import-not-found]
    _HAS_PYSTRAY = True
except ImportError:
    _HAS_PYSTRAY = False

from oss.capture.tray import config as cfg_mod
from oss.capture.tray import dll_inject
from oss.capture.tray import session_config
from oss.capture.tray import storage
from oss.capture.tray import steam_library
from oss.capture.tray import allowlist
from oss.capture.tray import process_watcher

log = logging.getLogger("oss.capture.tray.app")


CAPTURE_MODES = cfg_mod.CAPTURE_MODES


class TrayApp:
    """The OSS Capture tray-app controller.

    Holds the in-memory copy of the persistent config, the chosen output
    drive, and the pystray icon. The icon's menu items are bound to
    methods on this class.
    """

    def __init__(self) -> None:
        self.cfg = cfg_mod.load()
        self.icon: Optional["pystray.Icon"] = None
        self._janitor_stop = threading.Event()
        self._janitor_thread: Optional[threading.Thread] = None
        self._process_watcher: Optional[process_watcher.ProcessWatcher] = None

    # -----------------------------------------------------------------
    # Mode switcher
    # -----------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        """Persist a new capture mode. Live-effective on next game launch.

        v0 behavior: changing the mode while a game is already running
        does NOT update the in-game DLL (that requires IPC, scheduled for
        v1). Mode change here is picked up on the next launch.
        """
        if mode not in CAPTURE_MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {CAPTURE_MODES}")
        self.cfg.capture_mode = mode
        cfg_mod.save(self.cfg)
        log.info("capture mode set to %s", mode)
        if self.icon is not None:
            # Refresh the menu so the radio dot moves to the chosen entry.
            self.icon.update_menu()

    # -----------------------------------------------------------------
    # Pause toggle
    # -----------------------------------------------------------------

    def toggle_pause(self) -> None:
        self.cfg.paused = not self.cfg.paused
        cfg_mod.save(self.cfg)
        log.info("paused -> %s", self.cfg.paused)
        if self.icon is not None:
            self.icon.update_menu()

    # -----------------------------------------------------------------
    # Per-game enable toggles
    # -----------------------------------------------------------------

    def set_game_enabled(self, game_id: str, enabled: bool) -> None:
        self.cfg.enabled_games[game_id] = enabled
        cfg_mod.save(self.cfg)
        log.info("game %s enabled -> %s", game_id, enabled)
        if self.icon is not None:
            self.icon.update_menu()

    def toggle_game_enabled(self, game_id: str) -> None:
        self.set_game_enabled(game_id, not self.cfg.enabled_games.get(game_id, False))

    def _available_allowed_games(self) -> List[allowlist.AllowedGame]:
        games = []
        for installed in steam_library.all_installed_games():
            allowed = allowlist.lookup_by_app_id(installed.app_id)
            if allowed is not None:
                games.append(allowed)
        games.sort(key=lambda g: g.display_name.lower())
        return games

    # -----------------------------------------------------------------
    # Output drive resolution
    # -----------------------------------------------------------------

    def current_output_drive(self) -> Optional[str]:
        """Return the drive root currently in use (override or auto-pick)."""
        return storage.pick_output_drive(override=self.cfg.output_drive_override)

    def status_string(self) -> str:
        """Single-line status line for the tray menu."""
        drive = self.current_output_drive()
        if drive is None:
            drive_str = "no drive"
        else:
            try:
                free_gib = storage.list_candidate_drives([drive])[0].free_gib
                drive_str = f"{drive} ({free_gib:.0f} GiB free)"
            except (IndexError, OSError):
                drive_str = drive
        paused_str = " [PAUSED]" if self.cfg.paused else ""
        return f"OSS Capture: {self.cfg.capture_mode}, {drive_str}{paused_str}"

    # -----------------------------------------------------------------
    # Janitor (disk-cap background thread)
    # -----------------------------------------------------------------

    def _janitor_loop(self) -> None:
        """Periodically run cleanup_to_cap. Runs while the tray app is alive."""
        while not self._janitor_stop.is_set():
            drive = self.current_output_drive()
            if drive is not None:
                try:
                    storage.cleanup_to_cap(drive, self.cfg.disk_cap_bytes)
                except Exception:
                    log.exception("janitor sweep failed")
            # Run every 5 minutes. Captures don't accumulate fast enough to
            # need finer granularity, and the cleanup itself is cheap.
            self._janitor_stop.wait(timeout=300.0)

    def _start_janitor(self) -> None:
        if self._janitor_thread is not None:
            return
        self._janitor_thread = threading.Thread(
            target=self._janitor_loop, name="oss-capture-janitor", daemon=True,
        )
        self._janitor_thread.start()

    def _stop_janitor(self) -> None:
        self._janitor_stop.set()
        if self._janitor_thread is not None:
            self._janitor_thread.join(timeout=2.0)

    # -----------------------------------------------------------------
    # Process watcher + launch handling
    # -----------------------------------------------------------------

    def _start_process_watcher(self) -> None:
        if self._process_watcher is not None:
            return
        self._process_watcher = process_watcher.ProcessWatcher(
            on_game_launch=self._on_game_launch,
            on_anticheat_block=self._on_anticheat_block,
        )
        self._process_watcher.start()

    def _stop_process_watcher(self) -> None:
        if self._process_watcher is not None:
            self._process_watcher.stop()

    def _on_anticheat_block(self, event: process_watcher.AntiCheatBlockEvent) -> None:
        log.warning(
            "refusing to inject into %s pid=%d; blocking processes=%s",
            event.exe_basename,
            event.pid,
            ", ".join(event.blocking_processes),
        )

    def _on_game_launch(self, event: process_watcher.GameLaunchEvent) -> None:
        self.cfg = cfg_mod.load()
        game = event.allowed
        if self.cfg.paused:
            log.info("capture paused; skipping %s pid=%d", game.display_name, event.pid)
            return
        if not self.cfg.enabled_games.get(game.game_id, False):
            log.info("game %s is disabled; skipping pid=%d", game.game_id, event.pid)
            return

        drive = self.current_output_drive()
        if drive is None:
            log.warning("no output drive available; skipping %s pid=%d", game.game_id, event.pid)
            return

        output_dir = storage.captures_dir(drive) / game.game_id
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = session_config.write_session_config(
            game=game,
            capture_mode=self.cfg.capture_mode,
            output_dir=output_dir,
        )
        result = dll_inject.inject_dll(event.pid)
        if result.injected:
            log.info(
                "injected %s into %s pid=%d config=%s output=%s",
                result.dll_path,
                game.display_name,
                event.pid,
                config_path,
                output_dir,
            )
        elif result.skipped:
            log.info(
                "skipped DLL injection for %s pid=%d: %s",
                game.display_name,
                event.pid,
                result.message,
            )
        else:
            log.error(
                "DLL injection failed for %s pid=%d dll=%s: %s",
                game.display_name,
                event.pid,
                result.dll_path,
                result.message,
            )

    # -----------------------------------------------------------------
    # pystray menu construction
    # -----------------------------------------------------------------

    def _build_menu(self) -> "pystray.Menu":
        if not _HAS_PYSTRAY:
            raise RuntimeError("pystray not installed; tray UI is Windows-only")

        def _mode_item(mode: str) -> "pystray.MenuItem":
            return pystray.MenuItem(
                mode,
                lambda _icon, _item, m=mode: self.set_mode(m),
                radio=True,
                checked=lambda _item, m=mode: self.cfg.capture_mode == m,
            )

        def _game_item(game: allowlist.AllowedGame) -> "pystray.MenuItem":
            return pystray.MenuItem(
                game.display_name,
                lambda _icon, _item, gid=game.game_id: self.toggle_game_enabled(gid),
                checked=lambda _item, gid=game.game_id: self.cfg.enabled_games.get(gid, False),
            )

        games = self._available_allowed_games()
        games_menu = (
            pystray.Menu(*[_game_item(game) for game in games])
            if games
            else pystray.Menu(pystray.MenuItem("No allowlisted Steam games found", None, enabled=False))
        )

        return pystray.Menu(
            pystray.MenuItem(self.status_string, None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "Capture mode",
                pystray.Menu(*[_mode_item(m) for m in CAPTURE_MODES]),
            ),
            pystray.MenuItem("Enabled games", games_menu),
            pystray.MenuItem(
                lambda _item: ("Resume captures" if self.cfg.paused else "Pause captures"),
                lambda _icon, _item: self.toggle_pause(),
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "Open captures folder",
                lambda _icon, _item: self._open_captures_folder(),
            ),
            pystray.MenuItem(
                "Quit",
                lambda icon, _item: icon.stop(),
            ),
        )

    def _open_captures_folder(self) -> None:
        """Open the captures dir in Explorer (Windows) or Finder (macOS)."""
        drive = self.current_output_drive()
        if drive is None:
            log.warning("no output drive available; cannot open captures folder")
            return
        path = storage.captures_dir(drive)
        if platform.system() == "Windows":
            import subprocess
            subprocess.Popen(["explorer", str(path)])
        elif platform.system() == "Darwin":
            import subprocess
            subprocess.Popen(["open", str(path)])
        else:
            log.info("captures dir: %s", path)

    # -----------------------------------------------------------------
    # Icon image
    # -----------------------------------------------------------------

    @staticmethod
    def _make_icon_image() -> "Image.Image":
        """Generate a tiny tray-icon image (16x16 PNG of a stylized 'O').

        We don't have an artist; this draws a deterministic colored circle
        with a hole — the placeholder until a designed icon ships.
        """
        size = 16
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.ellipse((0, 0, size - 1, size - 1), fill=(40, 130, 255, 255))
        draw.ellipse((4, 4, size - 5, size - 5), fill=(0, 0, 0, 0))
        return img

    # -----------------------------------------------------------------
    # Run
    # -----------------------------------------------------------------

    def run(self) -> None:
        if not _HAS_PYSTRAY:
            print("pystray + Pillow are not installed.")
            print("Install on the Windows host with:")
            print("    pip install pystray pillow pywin32")
            sys.exit(1)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        self._start_janitor()
        self._start_process_watcher()

        self.icon = pystray.Icon(
            "oss-capture",
            icon=self._make_icon_image(),
            title="OSS Capture",
            menu=self._build_menu(),
        )

        try:
            self.icon.run()
        finally:
            self._stop_process_watcher()
            self._stop_janitor()


def main() -> int:
    TrayApp().run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
