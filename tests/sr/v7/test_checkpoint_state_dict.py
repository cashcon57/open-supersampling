from scripts.sr_train_v7 import _normalize_resume_model_state


def test_normalize_resume_model_state_strips_torch_compile_orig_mod_prefix():
    state = {
        "_orig_mod.root.weight": "root",
        "backbone._orig_mod.hat.patch_embed.weight": "w",
        "backbone._orig_mod.hat.patch_embed.bias": "b",
        "backbone._orig_mod.proj.weight": "p",
        "head.weight": "unchanged",
    }

    normalized = _normalize_resume_model_state(state)

    assert normalized == {
        "root.weight": "root",
        "backbone.hat.patch_embed.weight": "w",
        "backbone.hat.patch_embed.bias": "b",
        "backbone.proj.weight": "p",
        "head.weight": "unchanged",
    }


def test_normalize_resume_model_state_leaves_uncompiled_keys_untouched():
    state = {"backbone.hat.patch_embed.weight": "w", "head.weight": "h"}

    assert _normalize_resume_model_state(state) == state
