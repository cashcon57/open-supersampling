"""Thin S3-compatible wrapper for the Cloudflare R2 capture bucket.

R2 speaks the S3 API, so we use boto3 with a custom ``endpoint_url``.
Credentials are read from environment variables — never hardcoded:

- ``R2_ACCESS_KEY_ID``
- ``R2_SECRET_ACCESS_KEY``
- ``R2_ENDPOINT``
- ``R2_BUCKET``  (optional; defaults to ``ors-captures``)

In production the env is populated from ``.secrets/r2-credentials.env``
(gitignored). In tests we use ``moto`` to stand up a fake S3 endpoint and
override the constructor with explicit kwargs.

``boto3`` is a heavy import — about 200 ms cold — so it's lazy-imported
inside :class:`R2Client` rather than at module scope. This keeps
``python -m server.oss_capture_ingest.main --help`` snappy on a vanilla
Python without the package installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

from . import DEFAULT_R2_BUCKET


@dataclass
class R2Config:
    """All knobs for talking to R2.

    ``from_env`` reads the four standard env vars; tests construct one
    directly with the moto endpoint.
    """

    access_key_id: str
    secret_access_key: str
    endpoint_url: str
    bucket: str = DEFAULT_R2_BUCKET
    region_name: str = "auto"

    @classmethod
    def from_env(cls, *, env: Optional[Dict[str, str]] = None) -> "R2Config":
        e = env if env is not None else os.environ
        access = e.get("R2_ACCESS_KEY_ID", "")
        secret = e.get("R2_SECRET_ACCESS_KEY", "")
        endpoint = e.get("R2_ENDPOINT", "")
        bucket = e.get("R2_BUCKET") or DEFAULT_R2_BUCKET
        if not (access and secret and endpoint):
            raise RuntimeError(
                "R2 credentials missing — set R2_ACCESS_KEY_ID, "
                "R2_SECRET_ACCESS_KEY, R2_ENDPOINT (and optionally R2_BUCKET)"
            )
        return cls(
            access_key_id=access,
            secret_access_key=secret,
            endpoint_url=endpoint,
            bucket=bucket,
        )


# ---- key layout -------------------------------------------------------------


def month_partition(captured_at_unix: float) -> str:
    """Return ``YYYY-MM`` for a unix timestamp (UTC)."""
    dt = datetime.fromtimestamp(float(captured_at_unix), tz=timezone.utc)
    return dt.strftime("%Y-%m")


def frame_key(
    game_id: str,
    captured_at_unix: float,
    session_uuid: str,
    frame_uuid: str,
    *,
    suffix: str = ".exr",
) -> str:
    """Build the bucket key for a frame or its companion JSON.

    Layout matches the design memo:

        <game_id>/<YYYY-MM>/<session_uuid>/<frame_uuid>.{exr,json}
    """
    if suffix not in (".exr", ".json"):
        raise ValueError(f"unexpected suffix: {suffix!r}")
    month = month_partition(captured_at_unix)
    return f"{game_id}/{month}/{session_uuid}/{frame_uuid}{suffix}"


# ---- client -----------------------------------------------------------------


class R2Client:
    """Tiny S3-shaped wrapper, deliberately surface-area-minimal.

    We only need: ``put_object``, ``head_object`` (for dedup edge cases),
    ``list_objects_v2`` (for the daily index walker), and bucket creation
    fallback.
    """

    def __init__(self, config: R2Config, *, _client: Any = None) -> None:
        self.config = config
        # ``_client`` lets tests inject a moto-backed client without
        # dragging boto3 through a real HTTP signing dance.
        self._client_obj = _client

    # boto3 is lazy-imported here so module import is dependency-free
    def _client(self) -> Any:
        if self._client_obj is not None:
            return self._client_obj
        import boto3  # noqa: PLC0415 — intentional lazy import

        self._client_obj = boto3.client(
            "s3",
            endpoint_url=self.config.endpoint_url,
            aws_access_key_id=self.config.access_key_id,
            aws_secret_access_key=self.config.secret_access_key,
            region_name=self.config.region_name,
        )
        return self._client_obj

    # ---- ops --------------------------------------------------------------

    def ensure_bucket(self) -> None:
        """Create the bucket if it does not yet exist (idempotent)."""
        c = self._client()
        try:
            c.head_bucket(Bucket=self.config.bucket)
            return
        except Exception:
            # boto3's ClientError on 404; fall through to create.
            pass
        try:
            c.create_bucket(Bucket=self.config.bucket)
        except Exception as exc:  # pragma: no cover — depends on backend
            # Race or already-exists is fine; re-head once to confirm.
            try:
                c.head_bucket(Bucket=self.config.bucket)
            except Exception:
                raise exc

    def put_bytes(
        self,
        key: str,
        body: bytes,
        *,
        content_type: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        c = self._client()
        kwargs: Dict[str, Any] = {
            "Bucket": self.config.bucket,
            "Key": key,
            "Body": body,
            "ContentType": content_type,
        }
        if metadata:
            kwargs["Metadata"] = {str(k): str(v) for k, v in metadata.items()}
        c.put_object(**kwargs)

    def head(self, key: str) -> Optional[Dict[str, Any]]:
        c = self._client()
        try:
            return c.head_object(Bucket=self.config.bucket, Key=key)
        except Exception:
            return None

    def get_bytes(self, key: str) -> bytes:
        c = self._client()
        resp = c.get_object(Bucket=self.config.bucket, Key=key)
        return resp["Body"].read()

    def iter_objects(
        self, prefix: str = ""
    ) -> Iterator[Tuple[str, int]]:
        """Yield ``(key, size_bytes)`` for every object under ``prefix``."""
        c = self._client()
        token: Optional[str] = None
        while True:
            kwargs: Dict[str, Any] = {
                "Bucket": self.config.bucket,
                "Prefix": prefix,
            }
            if token is not None:
                kwargs["ContinuationToken"] = token
            resp = c.list_objects_v2(**kwargs)
            for obj in resp.get("Contents", []) or []:
                yield obj["Key"], int(obj.get("Size", 0))
            if resp.get("IsTruncated"):
                token = resp.get("NextContinuationToken")
                if token is None:
                    break
            else:
                break


def build_default_client(*, env: Optional[Dict[str, str]] = None) -> R2Client:
    """Construct an :class:`R2Client` from the standard env vars."""
    return R2Client(R2Config.from_env(env=env))
