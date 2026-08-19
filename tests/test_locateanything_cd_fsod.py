from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
import yaml

import ultralytics.models.locateanything.val as locate_val
from ultralytics.models.locateanything.model import LocateAnything
from ultralytics.models.locateanything.cd_fsod import (
    CD_FSOD_CLOSED_SET_PROTOCOL_ID,
    CD_FSOD_PROTOCOL_ID,
    CDFsodMetrics,
    build_category_alias_map,
    build_cd_fsod_validator,
    finalize_cd_fsod_results,
    load_cd_fsod_validation,
    naturalize_category_name,
)
from ultralytics.models.locateanything.val_preprocess import LocateAnythingValPreprocessor


def _write_dataset(
    tmp_path: Path,
    name: str,
    *,
    image_id: int | str,
    categories: list[dict],
    annotations: list[dict],
) -> Path:
    root = tmp_path / name
    (root / "test").mkdir(parents=True)
    (root / "annotations").mkdir()
    Image.new("RGB", (32, 24), "white").save(root / "test" / "image.jpg")
    payload = {
        "images": [{"id": image_id, "file_name": "image.jpg", "width": 32, "height": 24}],
        "annotations": annotations,
        "categories": categories,
    }
    (root / "annotations" / "test.json").write_text(json.dumps(payload), encoding="utf-8")
    config = tmp_path / f"{name}-1shot.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "train",
                "val": "test",
                "test": "test",
                "annotations": {"train": "annotations/1_shot.json", "val": "annotations/test.json"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config


def _suite_fixture(tmp_path: Path) -> list[Path]:
    artaxor = _write_dataset(
        tmp_path,
        "ArTaxOr",
        image_id="hash-image-id",
        categories=[{"id": 0, "name": "DUMMY_CLS"}, {"id": 1, "name": "rolled-in_scale"}],
        annotations=[{"id": 1, "image_id": "hash-image-id", "category_id": 1, "bbox": [1, 2, 8, 9]}],
    )
    uodd = _write_dataset(
        tmp_path,
        "UODD",
        image_id=7,
        categories=[{"id": 0, "name": "sea_urchin"}],
        annotations=[{"id": 2, "image_id": 7, "category_id": 0, "bbox": [2, 3, 7, 6]}],
    )
    return [artaxor, uodd]


def test_natural_category_aliases_keep_original_ids_and_reject_collisions():
    assert naturalize_category_name("rolled-in_scale") == "rolled in scale"
    categories = [{"id": 4, "name": "pitted_surface", "prompt_name": "pitted surface"}]
    aliases = build_category_alias_map(categories)
    assert aliases["pitted_surface"]["id"] == 4
    assert aliases["pitted surface"]["id"] == 4
    with pytest.raises(ValueError, match="类别别名冲突"):
        build_category_alias_map(
            [
                {"id": 1, "name": "a-b", "prompt_name": "a b"},
                {"id": 2, "name": "a_b", "prompt_name": "a b"},
            ]
        )


def test_cd_fsod_loader_supports_string_image_ids_category_zero_and_drops_dummy(tmp_path):
    suite = load_cd_fsod_validation(_suite_fixture(tmp_path))

    assert [dataset["id"] for dataset in suite["datasets"]] == ["ArTaxOr", "UODD"]
    assert [image["id"] for image in suite["images"]] == [1, 2]
    assert suite["images"][0]["source_image_id"] == "hash-image-id"
    assert suite["datasets"][0]["categories"][0]["name"] == "rolled-in_scale"
    assert suite["datasets"][0]["categories"][0]["prompt_name"] == "rolled in scale"
    assert suite["datasets"][1]["categories"][0]["source_category_id"] == 0
    assert all(category["name"] != "DUMMY_CLS" for category in suite["categories"])
    assert len({image["id"] for image in suite["images"]}) == 2


def test_cd_fsod_preprocessor_uses_natural_prompt_and_dedicated_protocol(tmp_path):
    suite = load_cd_fsod_validation(_suite_fixture(tmp_path))
    processor = LocateAnythingValPreprocessor(
        suite["annotations"],
        suite["categories"],
        category_aliases=suite["category_aliases"],
        protocol_id=CD_FSOD_PROTOCOL_ID,
    )

    resized, prompt, context = processor.prepare(suite["images"][0])

    assert resized.size == (1120, 840)
    assert prompt.endswith("rolled in scale.")
    assert context["validation_preprocess"]["protocol_id"] == CD_FSOD_PROTOCOL_ID
    assert processor.box_to_original([0, 0, 1120, 840], context)[2] == pytest.approx(1119 / 35)


def test_cd_fsod_closed_set_prompt_stays_inside_current_dataset(tmp_path):
    suite = load_cd_fsod_validation(_suite_fixture(tmp_path))
    processor = LocateAnythingValPreprocessor(
        suite["annotations"],
        suite["categories"],
        protocol="closed_set",
        category_aliases=suite["category_aliases"],
        protocol_id=CD_FSOD_CLOSED_SET_PROTOCOL_ID,
        category_ids_by_dataset=suite["category_ids_by_dataset"],
    )

    _, artaxor_prompt, artaxor = processor.prepare(suite["images"][0])
    _, uodd_prompt, uodd = processor.prepare(suite["images"][1])

    assert artaxor_prompt.endswith("rolled in scale.")
    assert "sea urchin" not in artaxor_prompt
    assert uodd_prompt.endswith("sea urchin.")
    assert "rolled in scale" not in uodd_prompt
    assert artaxor["validation_preprocess"]["protocol_id"] == CD_FSOD_CLOSED_SET_PROTOCOL_ID
    assert uodd["validation_preprocess"]["protocol_id"] == CD_FSOD_CLOSED_SET_PROTOCOL_ID


def test_cd_fsod_finalize_splits_datasets_and_computes_equal_weight_mean(tmp_path, monkeypatch):
    suite = load_cd_fsod_validation(_suite_fixture(tmp_path))
    records = []
    for image, dataset in zip(suite["images"], suite["datasets"]):
        category = dataset["categories"][0]
        annotation = dataset["annotations"][0]
        x, y, width, height = annotation["bbox"]
        records.append(
            {
                "image_id": image["id"],
                "source_image_id": image["source_image_id"],
                "dataset_id": image["dataset_id"],
                "file_name": image["file_name"],
                "raw_output": "",
                "parse_warnings": [],
                "unknown_labels": [],
                "predictions": [
                    {
                        "image_id": image["id"],
                        "category_id": category["id"],
                        "category_name": category["name"],
                        "bbox": [x, y, width, height],
                        "xyxy": [x, y, x + width, y + height],
                    }
                ],
                "speed": {"inference": 2.0},
                "output_tokens": 4,
                "generation_stats": {},
                "error": None,
            }
        )
    (tmp_path / "predictions.rank0.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    monkeypatch.setattr(
        locate_val,
        "run_nonstandard_coco_ap",
        lambda *args, **kwargs: {
            "status": "ok_nonstandard_constant_score",
            "AP50": 0.25,
            "AP50_95": 0.1,
            "AP75": 0.05,
        },
    )
    args = SimpleNamespace(
        model="nvidia/LocateAnything-3B",
        revision="fixed",
        devices="0",
        global_batch=1,
        batch=1,
        generation_mode="hybrid",
        max_new_tokens=8192,
        temperature=0.7,
        top_p=0.9,
        seed=0,
        protocol="paper",
    )

    payload = finalize_cd_fsod_results(args, tmp_path, suite, 1, 1.0, 2, 2, 8, 1.0, 1, 10)
    metrics = CDFsodMetrics(payload, tmp_path)

    assert metrics.results_dict["metrics/F1-mean(mean-datasets)"] == pytest.approx(1.0)
    assert payload["counts"] == {"datasets": 2, "images": 2, "boxes": 2, "ground_truths": 2}
    assert json.loads((tmp_path / "ArTaxOr" / "predictions.json").read_text())[0]["image_id"] == "hash-image-id"
    assert (tmp_path / "UODD" / "metrics.json").is_file()
    assert (tmp_path / "results.csv").is_file()

    args.protocol = "closed_set"
    closed_payload = finalize_cd_fsod_results(args, tmp_path, suite, 1, 1.0, 2, 2, 8, 1.0, 1, 10)
    closed_metrics = CDFsodMetrics(closed_payload, tmp_path)
    assert closed_payload["config"]["protocol_id"] == CD_FSOD_CLOSED_SET_PROTOCOL_ID
    assert closed_payload["datasets"]["ArTaxOr"]["auxiliary_paper_metrics"]["protocol"] == (
        "paper_style_on_closed_set_predictions"
    )
    assert closed_metrics.results_dict["metrics/closed-set-F1-mean(mean-datasets)"] == 1.0
    assert "paper_F1_mean" in (tmp_path / "results.csv").read_text().splitlines()[0]


def test_cd_fsod_validator_uses_one_suite_and_global_batch(tmp_path):
    data = _suite_fixture(tmp_path)
    owner = SimpleNamespace(
        model_name="nvidia/LocateAnything-3B",
        revision="fixed",
        npu_fast_path="auto",
    )
    validator = build_cd_fsod_validator(
        model=owner,
        data=data,
        device="0,1",
        batch=128,
        output_dir=tmp_path / "output",
        callbacks_={},
    )

    assert validator.args.benchmark == "cd_fsod"
    assert validator.args.total_images == 2
    assert validator.args.global_batch == 128
    assert validator.args.batch == 64
    assert validator.args.dataset_manifest_sha256
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    validator._prepare_run_config(output_dir)
    assert json.loads((output_dir / "run_config.json").read_text())["max_images_per_dataset"] == 0

    resumed = build_cd_fsod_validator(
        model=owner,
        data=data,
        device="0,1",
        batch=128,
        output_dir=output_dir,
        max_images_per_dataset=1,
        resume=True,
        callbacks_={},
    )
    with pytest.raises(RuntimeError, match="resume分布式布局不一致"):
        resumed._prepare_run_config(output_dir)


def test_cd_fsod_validator_accepts_closed_set_and_serializes_prompt_mode(tmp_path):
    owner = SimpleNamespace(
        model_name="nvidia/LocateAnything-3B",
        revision="fixed",
        npu_fast_path="auto",
    )
    output_dir = tmp_path / "closed"
    output_dir.mkdir()
    validator = build_cd_fsod_validator(
        model=owner,
        data=_suite_fixture(tmp_path),
        device="0,1",
        batch=128,
        output_dir=output_dir,
        protocol="closed_set",
        callbacks_={},
    )

    validator._prepare_run_config(output_dir)
    config = json.loads((output_dir / "run_config.json").read_text())
    assert validator.args.protocol == "closed_set"
    assert config["protocol"] == "closed_set"
    assert config["prompt_categories"] == "all_dataset_categories"


def test_model_val_cd_fsod_sets_and_returns_metrics(monkeypatch):
    expected = SimpleNamespace(results_dict={"fitness": 1.0})
    captured = {}

    class FakeValidator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __call__(self):
            return expected

    monkeypatch.setattr(
        "ultralytics.models.locateanything.cd_fsod.LocateAnythingCDFsodValidator",
        FakeValidator,
    )
    model = LocateAnything.__new__(LocateAnything)
    model.callbacks = {"on_val_start": []}

    result = model.val_cd_fsod(device="0,1", batch=128)

    assert result is expected
    assert model.metrics is expected
    assert captured["model"] is model
    assert captured["batch"] == 128
