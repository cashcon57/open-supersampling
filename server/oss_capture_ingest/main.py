"""FastAPI app factory + CLI entrypoint for the capture-ingest service.

CLI usage (no FastAPI required for ``--help``):

    python server/oss_capture_ingest/main.py --help
    python server/oss_capture_ingest/main.py serve --host 0.0.0.0 --port 8080
    python server/oss_capture_ingest/main.py mint-token --label "internal-dogfood"

The ``serve`` subcommand requires ``fastapi`` and ``uvicorn`` to be
installed; ``mint-token`` does not. This separation lets the per-game
installer build pipeline shell out to ``mint-token`` without dragging
the full server runtime in.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import uuid
from typing import Any, Optional, Sequence


log = logging.getLogger(__name__)


# ---- app factory (lazy fastapi import) -------------------------------------


def create_app(*, configure_r2_from_env: bool = True) -> Any:
    """Build and return the FastAPI application.

    ``fastapi`` is imported here, not at module top-level — that way
    ``import server.oss_capture_ingest.main`` works on a vanilla Python
    that lacks the dep, and ``--help`` still exits 0.
    """
    from fastapi import FastAPI  # noqa: PLC0415 — intentional lazy import
    from fastapi.middleware.cors import CORSMiddleware  # noqa: PLC0415
    from starlette.middleware.base import BaseHTTPMiddleware  # noqa: PLC0415

    from . import API_VERSION
    from .routes import ingest as ingest_routes
    from .routes import session as session_routes
    from .routes import stats as stats_routes

    app = FastAPI(
        title="OSS Capture Ingest",
        version=API_VERSION,
        description=(
            "Server-side ingestion API for the OSS Capture Tool. "
            "Accepts EXR + metadata uploads from the per-game installer's "
            "uploader daemon and writes them to the R2 ors-captures bucket."
        ),
    )

    # CORS: the uploader is a desktop daemon, not a browser, so we don't
    # strictly need this — but allow ``capture.oss-supersampling.dev`` and
    # localhost for the future contributor-stats web page.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://oss-supersampling.dev",
            "https://capture.oss-supersampling.dev",
            "http://localhost",
            "http://127.0.0.1",
        ],
        allow_origin_regex=r"^http://localhost(:\d+)?$",
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Request logging middleware — minimal, no PII (we don't have any).
    class RequestLogMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):  # type: ignore[override]
            t0 = time.time()
            response = await call_next(request)
            dt_ms = (time.time() - t0) * 1000.0
            log.info(
                "%s %s -> %d (%.1f ms)",
                request.method,
                request.url.path,
                response.status_code,
                dt_ms,
            )
            return response

    app.add_middleware(RequestLogMiddleware)

    # Routes — each module exposes ``build_router()`` to keep fastapi
    # imports lazy.
    app.include_router(ingest_routes.build_router())
    app.include_router(session_routes.build_router())
    app.include_router(stats_routes.build_router())

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok", "version": API_VERSION}

    # Optionally pre-resolve R2 from env so first /ingest doesn't pay the
    # boto3 import latency. Skipped for tests (which inject their own).
    if configure_r2_from_env:
        try:
            from .dedup import get_dedup  # noqa: PLC0415
            from .r2 import build_default_client  # noqa: PLC0415

            client = build_default_client()
            app.state.r2_client = client
            # Wire R2 as the durable backend for the dedup LRU. Closes
            # Codex's MED 'volatile dedup' finding — a process restart no
            # longer drops the dedup index.
            get_dedup().set_durable_backend(client)
        except RuntimeError as exc:
            log.warning(
                "R2 not configured at startup (%s); /ingest will return 500 "
                "until env vars are set",
                exc,
            )

    return app


# ---- CLI -------------------------------------------------------------------


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn  # noqa: PLC0415
    except ImportError:
        print(
            "uvicorn is required to serve. "
            "Install with: pip install 'fastapi>=0.115' 'uvicorn>=0.30'",
            file=sys.stderr,
        )
        return 2

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def _cmd_mint_token(args: argparse.Namespace) -> int:
    """Mint an install token + register it on the live process.

    Useful for ad-hoc dogfood: the installer build script also has a
    standalone copy of this so it can run without the full server runtime.
    """
    from .auth import get_registry  # noqa: PLC0415

    token = args.token or uuid.uuid4().hex
    registry = get_registry()
    registry.register_token(token, label=args.label or "")
    print(token)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oss-capture-ingest",
        description="OSS Capture Tool — server-side ingestion API.",
    )
    sub = parser.add_subparsers(dest="cmd")

    p_serve = sub.add_parser("serve", help="Run the FastAPI app via uvicorn.")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.add_argument("--log-level", default="info")
    p_serve.set_defaults(func=_cmd_serve)

    p_mint = sub.add_parser(
        "mint-token", help="Mint a new install token (for dogfood / testing)."
    )
    p_mint.add_argument("--label", default="", help="Human-readable label.")
    p_mint.add_argument(
        "--token",
        default=None,
        help="Use this exact token instead of generating a UUID4.",
    )
    p_mint.set_defaults(func=_cmd_mint_token)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
