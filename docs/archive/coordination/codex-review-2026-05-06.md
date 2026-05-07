# Codex review: v6 module audit 2026-05-06

Scope: audited every file under `oss/sr/v6/` except `model.py`, per request. Compared against `docs/superpowers/experiments/2026-05-05-v6-architecture-canonical.md` and `docs/research/2026-05-06-v6-source-extraction-notes.md`.

Test run:

`./venv-py312/bin/python -m pytest tests/sr/v6/ -q`

Result: 168 passed, 1 failed, 9 warnings. The failure is `tests/sr/v6/test_dataset.py::test_sr_train_v6_smoke_exits_on_v6model_stub`; `scripts/sr_train_v6.py` passes `warm_start` into excluded mid-write `oss/sr/v6/model.py`, and `V6Config.__init__()` rejects it.

Findings:

- HIGH `oss/sr/v6/losses.py:196`: `MultiScaleVGGLoss` casts inputs to fp32 but does not disable the caller's autocast context, so the frozen VGG convs still run with bf16 autocast outputs under bf16 training. Suggested fix: force the VGG submodule and inputs to fp32 and wrap both target and pred feature extraction in `with torch.autocast(device_type=pred.device.type, enabled=False)`.
- HIGH `oss/sr/v6/losses.py:315`: LPIPS-VGG receives fp32 tensors but is still executed inside the outer autocast context, so its pretrained VGG backbone runs bf16 internally. Suggested fix: call `lpips_module.float()` and wrap the LPIPS forward in `with torch.autocast(device_type=pred.device.type, enabled=False)`.
- HIGH `oss/sr/v6/aa_analytic_splat.py:131`: `torch.linalg.eigh` is used directly on projected covariance inverses; isotropic or repeated-eigenvalue inputs produce NaN gradients because the loss depends on arbitrary eigenvectors. Suggested fix: avoid eigenvector-dependent backward for 2x2 isotropic cases, or use a ridge/symmetrized closed-form 2x2 path with a stable fallback when eigenvalues are nearly equal.
- MEDIUM `oss/sr/v6/hat.py:303`: `hat_l()` does not match the HAT-L source-extraction note (`depths=[6]*12`, 72 HABs); it uses `depth=6, blocks_per_group=5` (30 HABs), so it cannot cleanly warm-start from HAT-L / GSASR-Ultra checkpoints. Suggested fix: either add an exact HAT-L factory for warm-start parity or rename this tier so code/docs stop claiming HAT-L compatibility.
- MEDIUM `oss/sr/v6/schedules.py:69`: the default scheduler builds `num_restarts + 1 == 4` cycles of `T_0=50K`, then returns lr=0 after step 200K, while the canonical training recipe runs 300K steps. Suggested fix: derive the cycle count from `max_steps=300_000` with 50K cycles, or set `T_0=75K` if the intended behavior is exactly three restarts over 300K.
- MEDIUM `oss/sr/v6/aa_perpendicular_dilation.py:32`: the AAA-Gaussians Eq. 10 adaptation dilates covariance but omits the required amplitude/opacity rescale `sqrt(|Sigma_perp| / |Sigma_hat_perp|)`, so splat energy changes after dilation. Suggested fix: return both `sigma_hat` and an opacity multiplier, computed with determinant/quadratic terms using ridge-regularized covariance solves.
- MEDIUM `oss/sr/v6/aa_view_space_angular.py:38`: this is documented as AAA-Gaussians Eqs. 14-17, but implements a screen-space `sqrt(diag(Sigma_2d))` AABB instead of view-space angular tangent bounds with `atan2` and epsilon clamping. Suggested fix: implement the source-note angular-bound equations in view space, or rename this helper as a conservative screen-space AABB fallback.
- MEDIUM `oss/sr/v6/aa_2dgs_object_space_mip.py:46`: the AA-2DGS object-space Mip filter ignores the source-note `nu_hat_k`, `s_reg`, and ray-splat Jacobian `J`; it reduces filtering to a scalar pixel-size ratio and erf box integral. Suggested fix: add per-splat sampling-frequency state and compute `V_k^eff = V_k + s_reg / nu_hat_k^2 * I_2`, then evaluate the Jacobian-based object-space filter.
- LOW `oss/sr/v6/dataset.py:123`: Hypersim still returns a zero `canvas_hint` channel even though the canonical memo says v6 drops the SRGD-era canvas hint and uses 9 channels total. Suggested fix: remove `canvas_hint` from v6 samples or keep it only behind a legacy adapter that cannot be concatenated into the v6 model input.
