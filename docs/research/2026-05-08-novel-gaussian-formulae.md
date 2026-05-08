# 2026-05-08 — Phase 4 Gaussian formulae record

## Status

No novelty claim. These are engineering derivations and known approximations applied to the OSS v6 rasterizer path.

## Formula 1: axis-aligned separability

For `rot=0`, `c=1`, `s=0`:

```math
a = c^2/s_x^2 + s^2/s_y^2 = 1/s_x^2
```

```math
b = cs(1/s_x^2 - 1/s_y^2) = 0
```

```math
d = s^2/s_x^2 + c^2/s_y^2 = 1/s_y^2
```

Then:

```math
q = a dx^2 + 2b dxdy + d dy^2 = dx^2/s_x^2 + dy^2/s_y^2
```

and:

```math
\exp(-q/2) = \exp(-dx^2/(2s_x^2)) \exp(-dy^2/(2s_y^2)).
```

Validity: exact for `rot=0`; also exact for `sx=sy` because rotation is irrelevant.

Error bound for dropping only the cross term over `|dx|<=3sx`, `|dy|<=3sy`:

```math
|\Delta w| \le 9 |\sin\theta\cos\theta| |s_x/s_y - s_y/s_x|.
```

Relation: standard separability of axis-aligned Gaussian kernels; used in EWA splatting and image filtering.

## Formula 2: 2D Gaussian q-tail mass

For isotropic 2D Gaussian radial/Mahalanobis variable `q=r^2/sigma^2`, the retained mass is:

```math
F(q) = 1 - \exp(-q/2).
```

Validity: isotropic radial or Mahalanobis ellipse in 2D.

Error bound: tail mass is exactly `exp(-q/2)`. At `q=12`, tail is `0.0024787521766663585`.

Relation: standard chi-square distribution with two degrees of freedom.

## Formula 3: LUT interpolation error for exp(-0.5q)

For `f(q)=exp(-q/2)` over a uniform grid with spacing `Delta q`:

```math
|f(q)-L(q)| \le \frac{\Delta q^2}{8}\max |f''(q)|
```

and `f''(q)=0.25 exp(-q/2)`, so over `q>=0`:

```math
|f(q)-L(q)| \le \Delta q^2 / 32.
```

For 256 entries on `[0,9]`, `Delta q=9/255`, bound is `3.8927335640138406e-5`.

Validity: linear interpolation, uniform table, exact table entries.

Relation: standard interpolation remainder theorem.

## Formula 4: diagonal Pade `[4/4]` for exp

```math
\exp(x) \approx
\frac{1 + x/2 + 3x^2/28 + x^3/84 + x^4/1680}
{1 - x/2 + 3x^2/28 - x^3/84 + x^4/1680}.
```

Validity tested here: `x=-0.5q`, `q in [0,9]`.

Measured max abs error: `5.833315084608631e-4`.

Relation: known diagonal Pade approximation to the exponential.

## Prior art

- `oss/gaussian/renderer/vendor/image_gs/` and `oss/gaussian/renderer/rasterizer.py`: project-local gsplat/Image-GS-derived raster path.
- Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, SIGGRAPH 2023.
- Zwicker et al., EWA splatting / surface splatting literature for Gaussian filtering and separable kernels.
- `docs/superpowers/experiments/2026-05-05-v6-architecture-canonical.md`: project record connecting v6 to 3DGS, GS-STVSR, 4DGS-1K, and GRAPE.
