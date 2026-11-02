import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asl_vit.data import build_index, label_maps, sample_and_split, save_label_mapping  # noqa: E402


@pytest.fixture
def fake_dataset(tmp_path):
    for label, n in {"B": 12, "A": 10, "space": 11, "del": 10}.items():
        d = tmp_path / label
        d.mkdir()
        for i in range(n):
            (d / f"{label}_{i}.jpg").write_bytes(b"x")
        (d / "notes.txt").write_text("ignored")
    (tmp_path / "README.md").write_text("not a class folder")
    return tmp_path


def test_index_lists_images_only(fake_dataset):
    df = build_index(str(fake_dataset))
    assert len(df) == 43
    assert set(df.label_name) == {"A", "B", "space", "del"}
    assert all(p.endswith(".jpg") for p in df.image_path)


def test_label_mapping_is_sorted_and_stable(fake_dataset):
    to_idx, to_label = label_maps(build_index(str(fake_dataset)))
    assert to_idx == {"A": 0, "B": 1, "del": 2, "space": 3}
    assert to_label[2] == "del"


def test_sample_and_split_is_balanced_stratified_and_deterministic(fake_dataset):
    df = build_index(str(fake_dataset))
    to_idx, _ = label_maps(df)
    tr1, va1 = sample_and_split(df, to_idx, per_class=10, val_fraction=0.2, seed=1)
    tr2, va2 = sample_and_split(df, to_idx, per_class=10, val_fraction=0.2, seed=1)
    assert len(tr1) == 32 and len(va1) == 8
    assert va1["label"].value_counts().to_dict() == {0: 2, 1: 2, 2: 2, 3: 2}
    assert set(tr1.image_path).isdisjoint(va1.image_path)
    assert tr1.equals(tr2) and va1.equals(va2)


def test_sample_rejects_classes_that_are_too_small(fake_dataset):
    df = build_index(str(fake_dataset))
    to_idx, _ = label_maps(df)
    with pytest.raises(ValueError):
        sample_and_split(df, to_idx, per_class=11)


def test_empty_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_index(str(tmp_path))


def test_label_mapping_file(tmp_path):
    save_label_mapping(tmp_path / "m.json", {"A": 0, "B": 1}, "vit")
    data = json.loads((tmp_path / "m.json").read_text())
    assert data["idx_to_label"] == {"0": "A", "1": "B"} and data["num_classes"] == 2
