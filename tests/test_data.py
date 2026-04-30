import numpy as np
import pyexr
import pytest

from ors.train.data import ORSDataset


@pytest.fixture
def fake_dataset(tmp_path):
    """One synthetic 8x8 image triplet on disk."""
    H, W = 8, 8
    base = "cbox_v0000"
    for k, c in [("noisy", 3), ("ground_truth", 3), ("albedo", 3),
                 ("normal", 3), ("depth", 1), ("motion", 2)]:
        arr = np.random.rand(H, W, c).astype(np.float32)
        pyexr.write(str(tmp_path / f"{base}_{k}.exr"), arr)
    return tmp_path


def test_dataset_loads_one_pair(fake_dataset):
    ds = ORSDataset(root=fake_dataset, augment=False)
    assert len(ds) == 1
    sample = ds[0]
    assert set(sample.keys()) >= {
        "noisy", "ground_truth", "aux", "history", "depth", "motion"
    }
    assert sample["aux"].shape[0] == 11
    assert sample["noisy"].shape[1:] == (8, 8)
