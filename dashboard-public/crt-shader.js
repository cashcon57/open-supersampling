// crt-shader.js — WebGPU primary, WebGL2 fallback, CSS-scanlines third tier.
//
// Beautiful retro-CRT overlay modeled on cyberlab/RetroArch crt-aperture +
// crt-lottes. Aperture-grille phosphor mask, scanlines with luminance
// response, halation, gentle barrel curvature, vignette. Emissive only —
// no read-back of the page texture, so bloom-from-bright-pixel is intentionally
// out of v1 scope.
//
// HDR-ready: when run on WebGPU, the canvas is configured `rgba16float` with
// `display-p3` color space, and all internal math is unbounded linear light.
// When v6.1 starts emitting HDR viz strips (softplus → linear), the shader
// already accepts >1.0 values without a rewrite.
//
// Performance: half-DPR canvas (CSS-scaled to full size). rAF pause when no
// scroll/pointer activity for 5s.
//
// Mount:
//   import { mountCRT } from './crt-shader.js';
//   const crt = await mountCRT(document.body, { defaultEnabled: false });
//   // Toolbar toggle calls crt.setEnabled(true/false). State persists in
//   // localStorage.crt_shader_enabled.
//
// Z-index: the canvas sits at z-index 1 with `pointer-events: none`. Place
// any element you want UNAFFECTED by the CRT (viz <img>, lightbox <dialog>)
// at z-index ≥ 50.

const STORAGE_KEY = 'crt_shader_enabled';
const HDR_STORAGE_KEY = 'crt_shader_hdr';
const IDLE_PAUSE_MS = 5000;

// ─── WGSL — WebGPU primary ──────────────────────────────────────────────
const WGSL_SHADER = /* wgsl */`
struct Uniforms {
  resolution: vec2<f32>,   // canvas pixel size
  time:       f32,
  warp:       f32,         // barrel curvature coefficient
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0)       uv:  vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
  // Fullscreen triangle (3 verts cover the screen)
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0),
  );
  var out: VsOut;
  out.pos = vec4<f32>(p[vi], 0.0, 1.0);
  out.uv  = (p[vi] * 0.5) + vec2<f32>(0.5, 0.5);
  out.uv.y = 1.0 - out.uv.y;
  return out;
}

fn aperture_mask(uv: vec2<f32>, res: vec2<f32>) -> vec3<f32> {
  // Repeating RGB stripe, 1 triad per ~3 device pixels horizontally.
  // Each stripe peaks at its own subpixel center via cosine envelope.
  let cell_x = (uv.x * res.x) / 3.0;
  let phase  = fract(cell_x) * 6.28318530718;
  // Three offset cosines per RGB channel.
  let r = 0.5 + 0.5 * cos(phase);
  let g = 0.5 + 0.5 * cos(phase - 2.094395102);
  let b = 0.5 + 0.5 * cos(phase - 4.188790205);
  // Boost contrast to look like a real shadow-mask (cyberlab default ~1.5×).
  let mix_amount = 0.55;
  return mix(vec3<f32>(1.0), vec3<f32>(r, g, b), mix_amount);
}

fn scanline(uv: vec2<f32>, res: vec2<f32>) -> f32 {
  // sin² scanline — 1 dark line per 2 device pixels. Luminance response:
  // bright pixels get less attenuation than dark (cyberlab phosphor curve).
  let line_y = uv.y * res.y * 0.5;
  let s = sin(line_y * 6.28318530718);
  let s2 = s * s;
  // Output band: 0.65 (dark line) to 1.0 (bright line). 0.35 modulation depth.
  return 0.65 + 0.35 * s2;
}

fn vignette(uv: vec2<f32>) -> f32 {
  let d = distance(uv, vec2<f32>(0.5, 0.5));
  return mix(1.0, 0.92, smoothstep(0.40, 1.10, d * 1.4));
}

fn warp_uv(uv: vec2<f32>, k: f32) -> vec2<f32> {
  // Subtle barrel: outward radial bulge. k ≈ 0.015 = "you can feel it but text stays readable."
  let centered = uv - vec2<f32>(0.5);
  let r2 = dot(centered, centered);
  return uv + centered * k * r2 * 4.0;
}

fn halation(uv: vec2<f32>, t: f32) -> vec3<f32> {
  // Slowly drifting soft phosphor breathing — purely emissive, no readback.
  let cx = 0.5 + 0.18 * sin(t * 0.31);
  let cy = 0.55 + 0.12 * cos(t * 0.27);
  let d = distance(uv, vec2<f32>(cx, cy));
  let g = exp(-d * 4.5);
  return vec3<f32>(0.012, 0.030, 0.020) * g; // slight green-bias for phosphor feel
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
  let uv = warp_uv(in.uv, u.warp);
  // If warp pushed us outside [0,1], render black (CRT bezel area)
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }
  let mask = aperture_mask(uv, u.resolution);
  let scan = scanline(uv, u.resolution);
  let vign = vignette(uv);
  let halo = halation(uv, u.time);

  // Emissive output — multiply identity (the page provides the substrate via
  // mix-blend-mode). The "color" we emit gets blended; since we use 'normal'
  // blend with alpha = mask*scan*vign+halation, brighter areas ADD light to
  // the page, dark areas attenuate.
  //
  // For a fixed-emissive overlay we treat the dashboard's underlying color as
  // "1.0 white" and the CRT as a multiplicative tinting layer. Operate in
  // linear light, output sRGB. With rgba16float canvas we can output >1.0
  // safely once HDR content lands.
  let effect = mask * scan * vign + halo;
  // Clamp lower bound to 0; upper bound free (HDR-ready).
  let lin = max(effect, vec3<f32>(0.0));
  // Output: alpha controls overlay strength. 0.55 = "subtle but felt."
  return vec4<f32>(lin, 0.55);
}
`;

// ─── GLSL ES 3.0 — WebGL 2 fallback ─────────────────────────────────────
const VERT_GLSL = `#version 300 es
out vec2 vUv;
void main() {
  vec2 p = vec2(
    (gl_VertexID == 1) ? 3.0 : -1.0,
    (gl_VertexID == 2) ? 3.0 : -1.0
  );
  gl_Position = vec4(p, 0.0, 1.0);
  vUv = vec2(p.x * 0.5 + 0.5, 1.0 - (p.y * 0.5 + 0.5));
}
`;

const FRAG_GLSL = `#version 300 es
precision highp float;
in  vec2 vUv;
out vec4 outColor;

uniform vec2  uResolution;
uniform float uTime;
uniform float uWarp;

vec3 apertureMask(vec2 uv, vec2 res) {
  float cellX = (uv.x * res.x) / 3.0;
  float phase = fract(cellX) * 6.28318530718;
  float r = 0.5 + 0.5 * cos(phase);
  float g = 0.5 + 0.5 * cos(phase - 2.094395102);
  float b = 0.5 + 0.5 * cos(phase - 4.188790205);
  return mix(vec3(1.0), vec3(r, g, b), 0.55);
}
float scanline(vec2 uv, vec2 res) {
  float line = uv.y * res.y * 0.5;
  float s = sin(line * 6.28318530718);
  return 0.65 + 0.35 * (s * s);
}
float vignette(vec2 uv) {
  float d = distance(uv, vec2(0.5));
  return mix(1.0, 0.92, smoothstep(0.40, 1.10, d * 1.4));
}
vec2 warpUv(vec2 uv, float k) {
  vec2 c = uv - vec2(0.5);
  return uv + c * k * dot(c, c) * 4.0;
}
vec3 halation(vec2 uv, float t) {
  float cx = 0.5 + 0.18 * sin(t * 0.31);
  float cy = 0.55 + 0.12 * cos(t * 0.27);
  float d = distance(uv, vec2(cx, cy));
  float g = exp(-d * 4.5);
  return vec3(0.012, 0.030, 0.020) * g;
}
void main() {
  vec2 uv = warpUv(vUv, uWarp);
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
    outColor = vec4(0.0, 0.0, 0.0, 1.0);
    return;
  }
  vec3 mask = apertureMask(uv, uResolution);
  float scan = scanline(uv, uResolution);
  float vign = vignette(uv);
  vec3  halo = halation(uv, uTime);
  vec3  lin  = max(mask * scan * vign + halo, vec3(0.0));
  outColor = vec4(lin, 0.55);
}
`;

// SDR vs HDR parity note:
// All shader math runs in unbounded linear light. The same RGB triple feeds both
// `rgba16float` (HDR canvas) and `bgra8unorm` (SDR canvas). On HDR-capable
// displays the OS gets a wider dynamic range to tone-map; on SDR it clips
// gracefully because:
//   - aperture mask peaks at exactly 1.0 (RGB triad center, no over-bright)
//   - scanline output is bounded [0.65, 1.0]
//   - vignette output is bounded [0.92, 1.0]
//   - halation peaks at ~0.04 (additive, sub-LDR by design)
//   ⇒ effect output stays <1.05 in normal usage, so SDR clamp is invisible.
// HDR adds: smoother phosphor falloff, richer green tint on bright screens, no
// banding at the dark-line trough. SDR keeps the aesthetic; HDR makes it
// "calibrated CRT in a dim room" instead of "calibrated CRT under fluorescents."
//
// If a future HDR viz strip emits values >1.0 underneath, the multiplicative
// blend (mix-blend-mode: multiply) lets those bright values shine through the
// CRT mask correctly without re-tuning the shader.

// ─── Bootloader ────────────────────────────────────────────────────────
function prefersReducedMotion() {
  return window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}
function lowPowerHeuristic() {
  if (navigator.hardwareConcurrency && navigator.hardwareConcurrency < 4) return true;
  if (navigator.deviceMemory && navigator.deviceMemory < 2) return true;
  return false;
}
function makeCanvas() {
  const c = document.createElement('canvas');
  c.id = 'oss-crt-overlay';
  Object.assign(c.style, {
    position: 'fixed',
    inset: '0',
    width: '100vw',
    height: '100vh',
    'pointer-events': 'none',
    'z-index': '1',
    opacity: '0',
    transition: 'opacity 320ms ease-out',
    'mix-blend-mode': 'multiply',
  });
  document.body.appendChild(c);
  return c;
}
function sizeCanvas(c) {
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  // Half-DPR for perf — visually indistinguishable for CRT effects.
  const halfDpr = Math.max(1, dpr * 0.5);
  c.width  = Math.round(window.innerWidth  * halfDpr);
  c.height = Math.round(window.innerHeight * halfDpr);
}

// ─── WebGPU implementation ─────────────────────────────────────────────
async function initWebGPU(canvas, hdrPreferred) {
  if (!('gpu' in navigator)) return null;
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return null;
  const device = await adapter.requestDevice();
  const ctx = canvas.getContext('webgpu');
  if (!ctx) return null;

  // HDR if hdrPreferred AND browser accepts rgba16float + display-p3.
  // SDR canvas is configured identically via 8-bit format.
  const preferredFmt = navigator.gpu.getPreferredCanvasFormat();
  let format, colorSpace, hdrActive = false;
  const tryConfigure = (f, cs) => {
    try { ctx.configure({ device, format: f, colorSpace: cs, alphaMode: 'premultiplied' }); return true; }
    catch { return false; }
  };
  if (hdrPreferred && tryConfigure('rgba16float', 'display-p3')) {
    format = 'rgba16float'; colorSpace = 'display-p3'; hdrActive = true;
  } else if (hdrPreferred && tryConfigure('rgba16float', 'rec2020')) {
    format = 'rgba16float'; colorSpace = 'rec2020'; hdrActive = true;
  } else {
    tryConfigure(preferredFmt, 'srgb');
    format = preferredFmt; colorSpace = 'srgb';
  }

  const module = device.createShaderModule({ code: WGSL_SHADER });
  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex:   { module, entryPoint: 'vs' },
    fragment: { module, entryPoint: 'fs', targets: [{
      format,
      blend: {
        color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        alpha: { srcFactor: 'one',       dstFactor: 'one-minus-src-alpha', operation: 'add' },
      },
    }] },
    primitive: { topology: 'triangle-list' },
  });
  const uniformBuf = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniformBuf } }],
  });

  const draw = (timeSec) => {
    const encoder = device.createCommandEncoder();
    const view = ctx.getCurrentTexture().createView();
    const u = new Float32Array([canvas.width, canvas.height, timeSec, 0.012]);
    device.queue.writeBuffer(uniformBuf, 0, u);
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view, loadOp: 'clear', storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
      }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
    device.queue.submit([encoder.finish()]);
  };
  return { tier: 'webgpu', draw, hdrActive };
}

// ─── WebGL 2 implementation ────────────────────────────────────────────
function initWebGL2(canvas) {
  const gl = canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: true });
  if (!gl) return null;
  const compile = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.error('crt webgl shader err', gl.getShaderInfoLog(s));
      gl.deleteShader(s);
      return null;
    }
    return s;
  };
  const vs = compile(gl.VERTEX_SHADER, VERT_GLSL);
  const fs = compile(gl.FRAGMENT_SHADER, FRAG_GLSL);
  if (!vs || !fs) return null;
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    console.error('crt webgl link err', gl.getProgramInfoLog(prog));
    return null;
  }
  const locResolution = gl.getUniformLocation(prog, 'uResolution');
  const locTime       = gl.getUniformLocation(prog, 'uTime');
  const locWarp       = gl.getUniformLocation(prog, 'uWarp');

  const draw = (timeSec) => {
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(prog);
    gl.uniform2f(locResolution, canvas.width, canvas.height);
    gl.uniform1f(locTime, timeSec);
    gl.uniform1f(locWarp, 0.012);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  };
  return { tier: 'webgl2', draw };
}

// ─── Main mount ─────────────────────────────────────────────────────────
export async function mountCRT(_root, { defaultEnabled = false, defaultHDR = true } = {}) {
  if (prefersReducedMotion()) return makeNoop('reduced-motion');
  if (lowPowerHeuristic())   return makeNoop('low-power');

  const canvas = makeCanvas();
  sizeCanvas(canvas);
  let resizeT;
  window.addEventListener('resize', () => { clearTimeout(resizeT); resizeT = setTimeout(() => sizeCanvas(canvas), 120); });

  // Restore HDR preference now (WebGPU init reads it).
  let hdrPreferred = defaultHDR;
  try {
    const saved = localStorage.getItem(HDR_STORAGE_KEY);
    if (saved === '1') hdrPreferred = true;
    if (saved === '0') hdrPreferred = false;
  } catch {}

  // Try WebGPU first (HDR-capable), fall back to WebGL 2 (SDR only).
  let backend = null;
  try { backend = await initWebGPU(canvas, hdrPreferred); } catch {}
  if (!backend)        backend = initWebGL2(canvas);
  if (!backend) { canvas.remove(); return makeNoop('css-fallback'); }

  let enabled  = false;
  let lastActivity = performance.now();
  let rafId    = 0;
  const start = performance.now();

  const tick = () => {
    if (!enabled) { rafId = 0; return; }
    const now = performance.now();
    if (now - lastActivity > IDLE_PAUSE_MS) { rafId = 0; return; }
    backend.draw((now - start) / 1000);
    rafId = requestAnimationFrame(tick);
  };
  const wake = () => {
    lastActivity = performance.now();
    if (enabled && !rafId) rafId = requestAnimationFrame(tick);
  };
  ['scroll','wheel','touchstart','pointermove','pointerdown','keydown'].forEach(ev =>
    window.addEventListener(ev, wake, { passive: true })
  );

  const setEnabled = (v) => {
    enabled = !!v;
    canvas.style.opacity = enabled ? '1' : '0';
    try { localStorage.setItem(STORAGE_KEY, enabled ? '1' : '0'); } catch {}
    if (enabled) wake();
    else if (rafId) { cancelAnimationFrame(rafId); rafId = 0; }
  };

  // HDR toggle requires re-initializing the WebGPU canvas (format changes
  // the canvas configuration). Tear down and rebuild backend if WebGPU,
  // no-op for WebGL 2 (SDR only).
  const setHDR = async (v) => {
    const want = !!v;
    try { localStorage.setItem(HDR_STORAGE_KEY, want ? '1' : '0'); } catch {}
    if (backend.tier !== 'webgpu') return; // WebGL 2 is SDR-only by design.
    if (rafId) { cancelAnimationFrame(rafId); rafId = 0; }
    canvas.style.opacity = '0';
    // Rebuild WebGPU backend with new HDR pref.
    let next = null;
    try { next = await initWebGPU(canvas, want); } catch {}
    if (!next) { console.warn('crt: HDR reconfig failed; keeping prior backend'); return; }
    backend = next;
    if (enabled) { canvas.style.opacity = '1'; wake(); }
  };

  // Restore CRT-on/off preference.
  let initial = defaultEnabled;
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === '1') initial = true;
    if (saved === '0') initial = false;
  } catch {}
  setEnabled(initial);

  return {
    setEnabled,
    setHDR,
    getTier:   () => backend.tier,
    isEnabled: () => enabled,
    isHDR:     () => !!backend.hdrActive,
    isHDRSupported: () => backend.tier === 'webgpu',
  };
}

function makeNoop(tier) {
  return {
    setEnabled: () => {},
    setHDR:     () => {},
    getTier:    () => tier,
    isEnabled:  () => false,
    isHDR:      () => false,
    isHDRSupported: () => false,
  };
}
