// oss-dashboard-uploader — Cloudflare Worker (deployed at upload.opensupersampling.com
// + opensupersampling.com / opensupersampling.org via Workers Custom Domains).
//
// THIS IS THE LIVE DEPLOYED SOURCE. It's checked in for transparency + audit
// — the actual deploy lives on Cloudflare's edge, written via the cloudflare-api
// MCP from the maintainer's session. To redeploy after edits here, push to the
// `oss-dashboard-uploader` script via PUT /accounts/<id>/workers/scripts/<name>
// with this file as the `worker.js` part of a multipart/form-data body. The
// `BUCKET` binding (R2 oss-dashboard) and `SHARED_SECRET` (env secret) are set
// out of band and survive script updates because metadata.keep_secrets = true.
//
// Hardened build per the 2026-05-07 security review:
//   - Constant-time auth (bitwise XOR over equal-length bytes)
//   - PUT body-size cap (8 MiB; rejects 413 if larger or 411 if missing CL)
//   - Key-prefix allow-list (no arbitrary key namespaces can be created)
//   - DELETE removed entirely (was unused; eliminates wipe-everything risk)
//
// Reads (GET / HEAD) are unauthenticated and serve from R2 directly. HTML and
// JSON are no-cache-must-revalidate so the dashboard always sees fresh data;
// other assets get a 30s edge cache.

const MAX_PUT_BYTES = 8 * 1024 * 1024;  // 8 MiB

const KEY_PREFIX_ALLOWLIST = [
  'data.json',
  'status.json',
  'index.html',
  'oss-logo.svg',
  'crt-shader.js',
  'runs/',                  // any per-run files
  'test/',                  // smoke / health tests
];

function keyAllowed(key) {
  for (const allowed of KEY_PREFIX_ALLOWLIST) {
    if (allowed.endsWith('/')) {
      if (key.startsWith(allowed)) return true;
    } else if (key === allowed) {
      return true;
    }
  }
  return false;
}

function timingSafeEqual(a, b) {
  if (typeof a !== 'string' || typeof b !== 'string') return false;
  if (a.length !== b.length) return false;
  const ae = new TextEncoder().encode(a);
  const be = new TextEncoder().encode(b);
  let diff = 0;
  for (let i = 0; i < ae.length; i++) diff |= ae[i] ^ be[i];
  return diff === 0;
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname;
    const method = request.method;

    if (method === 'GET' && path === '/health') {
      return new Response(JSON.stringify({ ok: true, ts: Date.now() }), {
        headers: { 'content-type': 'application/json', 'cache-control': 'no-store' }
      });
    }

    if (method === 'GET' || method === 'HEAD') {
      if (path === '/health') {
        return new Response(null, { status: 200, headers: { 'cache-control': 'no-store' } });
      }
      const key = path === '/' ? 'index.html' : path.slice(1);
      const obj = await env.BUCKET.get(key);
      if (!obj) return new Response('not found', { status: 404, headers: { 'cache-control': 'no-store' } });
      const headers = new Headers();
      if (obj.writeHttpMetadata) obj.writeHttpMetadata(headers);
      headers.set('etag', obj.httpEtag);
      if (key === 'index.html' || key.endsWith('.html') || key === 'data.json' || key.endsWith('.json')) {
        headers.set('cache-control', 'no-cache, must-revalidate');
      } else {
        headers.set('cache-control', 'public, max-age=30, s-maxage=30');
      }
      headers.set('x-served-by', 'oss-dashboard-uploader');
      if (method === 'HEAD') return new Response(null, { headers });
      return new Response(obj.body, { headers });
    }

    // Auth (constant-time)
    const auth = request.headers.get('authorization') || '';
    const expected = 'Bearer ' + env.SHARED_SECRET;
    if (!timingSafeEqual(auth, expected)) {
      return new Response('unauthorized', { status: 401, headers: { 'cache-control': 'no-store' } });
    }

    if (method === 'PUT' && path.startsWith('/upload/')) {
      const key = path.slice('/upload/'.length);
      if (!key) return new Response('missing key', { status: 400 });
      if (!keyAllowed(key)) {
        return new Response('key not in allow-list', { status: 403, headers: { 'cache-control': 'no-store' } });
      }
      const len = Number(request.headers.get('content-length') || 'NaN');
      if (!Number.isFinite(len)) return new Response('content-length required', { status: 411 });
      if (len > MAX_PUT_BYTES) return new Response('payload too large', { status: 413 });
      const ctype = request.headers.get('content-type') || 'application/octet-stream';
      await env.BUCKET.put(key, request.body, { httpMetadata: { contentType: ctype } });
      return new Response(JSON.stringify({ ok: true, key }), { status: 200, headers: { 'content-type': 'application/json' } });
    }

    // DELETE intentionally removed — not used by the dashboard pipeline. If
    // future use needs it, re-add with `X-OSS-Confirm-Delete: yes` required.

    return new Response('not found', { status: 404, headers: { 'cache-control': 'no-store' } });
  },
};
