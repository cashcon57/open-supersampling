# PyInstaller spec for the OSS capture uploader.
#
# Build from the repo root on Windows:
#   pyinstaller oss/capture/installer/oss_capture_uploader_pyinstaller.spec

from PyInstaller.utils.hooks import collect_submodules


block_cipher = None

a = Analysis(
    ["oss/capture/uploader.py"],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=collect_submodules("oss.capture"),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch", "torchvision", "cv2", "numpy", "onnx", "onnxruntime", "mitsuba"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="oss_capture_uploader",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

