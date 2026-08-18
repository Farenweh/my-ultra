from __future__ import annotations

import json
from argparse import Namespace
from collections import defaultdict
from types import SimpleNamespace

import pytest
import numpy as np
import torch
from PIL import Image

import ultralytics.models.locateanything.batch as locate_batch
import ultralytics.models.locateanything.val as locate_val
from ultralytics.models.locateanything.batch import (
    BatchInput,
    DeviceTokenHistory,
    ExpandableStaticKVCache,
    PagedKVCache,
    QSampleReservoir,
    _apply_repetition_penalty,
    _candidate_top_p_values,
    _decode_mtp_tokens,
    _encode_visual_features,
    _fill_q_sample_buffer,
    _forward_decode,
    _forward_paged_decoder,
    _guard_duplicate_box_pattern,
    _handle_pattern_tokens,
    _paged_right_down_causal_mask,
    _sample_parameter_tensors,
    _sample_probabilities,
    _step_mixed_paged,
    _summarize_mtp_distribution,
    generate_batch_hybrid,
    make_row_generators,
    pack_kv_rows,
    unpack_kv_row,
)
from ultralytics.models.locateanything.model import LocateAnything, _ContinuousBatchPrefetcher
from ultralytics.models.locateanything.val_preprocess import (
    LEGACY_PROTOCOL_ID,
    PAPER_PROTOCOL_ID,
    PAPER_SHORT_SIDE,
    LocateAnythingValPreprocessor,
)


def _prediction(image_id: int, xyxy: list[float], category_id: int = 1) -> dict:
    return {
        "image_id": image_id,
        "category_id": category_id,
        "category_name": "person",
        "bbox": locate_val.xyxy_to_xywh(xyxy),
        "xyxy": xyxy,
    }


def _record(image_id: int, predictions: list[dict]) -> dict:
    return {
        "image_id": image_id,
        "file_name": f"{image_id}.jpg",
        "raw_output": "",
        "parse_warnings": [],
        "unknown_labels": [],
        "predictions": predictions,
        "speed": {"inference": 10.0},
        "error": None,
    }


@pytest.mark.parametrize(
    ("size", "resized_size"),
    [((640, 480), (1120, 840)), ((480, 640), (840, 1120))],
)
def test_paper_val_preprocessor_resizes_short_side_and_builds_positive_prompt(tmp_path, size, resized_size):
    source = Image.effect_noise(size, 32).convert("RGB")
    path = tmp_path / "image.png"
    source.save(path)
    image = {"id": 7, "path": str(path), "file_name": path.name, "width": size[0], "height": size[1]}
    processor = LocateAnythingValPreprocessor(
        [{"image_id": 7, "category_id": 3}, {"image_id": 7, "category_id": 1}],
        [{"id": 3, "name": "car"}, {"id": 1, "name": "person"}],
    )

    resized, question, context = processor.prepare(image)

    assert resized.size == resized_size
    assert question.endswith("person</c>car.")
    metadata = context["validation_preprocess"]
    assert metadata["protocol_id"] == PAPER_PROTOCOL_ID
    assert metadata["short_side"] == PAPER_SHORT_SIDE
    assert metadata["interpolation"] == "bilinear"
    assert metadata["original_size"] == list(size)
    assert metadata["resized_size"] == list(resized_size)
    expected = source.resize(resized_size, Image.Resampling.BILINEAR)
    assert np.array_equal(np.asarray(resized), np.asarray(expected))


def test_paper_val_preprocessor_maps_clipped_resized_box_back_to_original(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (640, 480)).save(path)
    image = {"id": 1, "path": str(path), "file_name": path.name, "width": 640, "height": 480}
    processor = LocateAnythingValPreprocessor(
        [{"image_id": 1, "category_id": 1}],
        [{"id": 1, "name": "person"}],
    )
    _, _, context = processor.prepare(image)
    mapped = processor.box_to_original([-1, -2, 1120, 840], context)
    assert mapped == pytest.approx([0.0, 0.0, 1119 / 1.75, 839 / 1.75])


def test_paper_val_preprocessor_keeps_empty_prompt_for_image_without_gt_category(tmp_path):
    processor = LocateAnythingValPreprocessor([], [{"id": 1, "name": "person"}])
    path = tmp_path / "empty.png"
    Image.new("RGB", (16, 12)).save(path)
    image = {"id": 9, "path": str(path), "file_name": path.name, "width": 16, "height": 12}
    _, question, context = processor.prepare(image)
    assert question == "Locate all the instances that matches the following description: ."
    assert context["validation_preprocess"]["prompt_categories"] == []


def test_legacy_val_preprocessor_keeps_original_path_and_all_categories(tmp_path):
    image = {"id": 1, "path": str(tmp_path / "image.jpg"), "file_name": "image.jpg", "width": 8, "height": 6}
    processor = LocateAnythingValPreprocessor(
        [{"image_id": 1, "category_id": 1}],
        [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}],
        protocol="legacy",
    )
    source, question, context = processor.prepare(image)
    assert source == image["path"]
    assert question.endswith("person</c>car.")
    assert context["validation_preprocess"]["protocol_id"] == LEGACY_PROTOCOL_ID


def test_eight_rank_stride_sharding_has_exact_coverage():
    images = [{"id": index} for index in range(37)]
    shards = [locate_val.shard_images(images, rank, 8) for rank in range(8)]
    flattened = [item["id"] for shard in shards for item in shard]
    assert sorted(flattened) == list(range(37))
    assert len(flattened) == len(set(flattened))
    assert [item["id"] for item in shards[3]] == [3, 11, 19, 27, 35]


def test_batch_groups_keep_requested_size_and_real_tail():
    images = [{"id": index} for index in range(19)]
    groups = locate_val.batch_images(images, 8)
    assert [len(group) for group in groups] == [8, 8, 3]
    assert [item["id"] for group in groups for item in group] == list(range(19))


def test_dynamic_queue_claims_each_image_once_and_resume_skips_success(tmp_path):
    images = [{"id": index} for index in range(10)]
    success = _record(2, [])
    failed = _record(3, [])
    failed["error"] = "失败"
    (tmp_path / "predictions.rank0.jsonl").write_text(json.dumps(success) + "\n", encoding="utf-8")
    (tmp_path / "predictions.rank1.jsonl").write_text(json.dumps(failed) + "\n", encoding="utf-8")
    queued = locate_val._initialize_dynamic_queue(tmp_path, images, 2, resume=True)
    assert queued == [0, 1, 3, 4, 5, 6, 7, 8, 9]

    first = locate_val._DynamicImageQueue(tmp_path)
    second = locate_val._DynamicImageQueue(tmp_path)
    claimed = first.claim(3) + second.claim(4) + first.claim(8)
    assert claimed == queued
    assert len(claimed) == len(set(claimed))
    assert second.claim(1) == []


def test_tcp_dynamic_queue_reserves_initial_local_batch_per_rank(monkeypatch):
    class AtomicStore:
        def __init__(self):
            self.values = {}

        def set(self, key, value):
            self.values[key] = int(value)

        def add(self, key, value):
            self.values[key] += int(value)
            return self.values[key]

    store = AtomicStore()
    image_ids = list(range(12))
    monkeypatch.setattr(locate_val.dist, "barrier", lambda: None)
    rank0 = locate_val._TCPDynamicImageQueue(store, image_ids, "reserved", 0, 2, 3)
    rank1 = locate_val._TCPDynamicImageQueue(store, image_ids, "reserved", 1, 2, 3)

    assert rank0.claim(3) == [0, 1, 2]
    assert rank1.claim(3) == [3, 4, 5]
    assert rank0.claim(2) == [6, 7]


def test_continuous_prefetcher_preserves_order_and_stops_at_total():
    pending = [(index, index + 10, {"id": index}) for index in range(5)]
    requests = []

    def provider(count):
        requests.append(count)
        items = pending[:count]
        del pending[:count]
        return items

    def prepare(source, question):
        assert question == "问题"
        return source * 2, (np.zeros((1, 1, 3)), str(source)), float(source)

    prefetcher = _ContinuousBatchPrefetcher(
        provider,
        prepare,
        "问题",
        total=5,
        request_size=2,
        capacity=2,
    )
    try:
        first = prefetcher.get(3)
        second = prefetcher.get(3)
    finally:
        prefetcher.close()
    assert [item[0] for item in first + second] == [0, 2, 4, 6, 8]
    assert [item[3] for item in first + second] == [10, 11, 12, 13, 14]
    assert requests == [2, 2, 1]


def test_continuous_prefetcher_accepts_per_sample_questions():
    pending = [(1, 11, {"id": 1}, "person"), (2, 12, {"id": 2}, "car")]
    seen = []

    def provider(count):
        items = pending[:count]
        del pending[:count]
        return items

    def prepare(source, question):
        seen.append((source, question))
        return source, (np.zeros((1, 1, 3)), str(source)), 0.0

    prefetcher = _ContinuousBatchPrefetcher(provider, prepare, None, total=2, request_size=2, capacity=2)
    try:
        items = prefetcher.get(2)
    finally:
        prefetcher.close()
    assert [item[3] for item in items] == [11, 12]
    assert seen == [(1, "person"), (2, "car")]


def test_async_record_sink_flushes_all_records_in_order(tmp_path):
    path = tmp_path / "predictions.jsonl"
    with locate_val._RecordSink(path, "w", asynchronous=True) as sink:
        for image_id in range(20):
            sink.write({"image_id": image_id})
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["image_id"] for record in records] == list(range(20))


def test_device_list_accepts_any_positive_unique_npu_count():
    assert locate_val.parse_devices("0") == [0]
    assert locate_val.parse_devices("0,1") == [0, 1]
    assert locate_val.parse_devices("0,1,2,3,4,5,6,7") == list(range(8))
    with pytest.raises(ValueError, match="至少一个互不重复"):
        locate_val.parse_devices("")
    with pytest.raises(ValueError, match="至少一个互不重复"):
        locate_val.parse_devices("0,1,2,3,4,5,6,6")


def test_only_nonzero_rank_disables_transformers_progress(monkeypatch):
    from transformers.utils import logging as transformers_logging

    calls = []
    monkeypatch.setattr(transformers_logging, "disable_progress_bar", lambda: calls.append("disabled"))
    locate_val._configure_worker_progress(0)
    locate_val._configure_worker_progress(1)
    assert calls == ["disabled"]


def test_worker_cpu_runtime_uses_npu_topology(monkeypatch):
    calls = {}
    topology = "NPU0 X HCCS 144-167\nNPU1 HCCS X 144-167\n"
    monkeypatch.setattr(locate_val.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=topology))
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda value: calls.update(threads=value))
    monkeypatch.setattr(locate_val.os, "sched_setaffinity", lambda pid, cpus: calls.update(pid=pid, cpus=cpus))
    assert locate_val._configure_worker_cpu_runtime(0) == set(range(144, 168))
    assert calls == {"threads": 1, "pid": 0, "cpus": set(range(144, 168))}
    assert locate_val._parse_cpu_list("0-3,8,10-11") == {0, 1, 2, 3, 8, 10, 11}


def test_label_mapping_and_fake_result_schema():
    known = SimpleNamespace(label=" Person。 ", xyxy=(1.0, 2.0, 11.0, 22.0))
    unknown = SimpleNamespace(label="human", xyxy=(2.0, 2.0, 5.0, 5.0))
    result = SimpleNamespace(
        boxes=[known, unknown],
        raw_output="<ref>Person。</ref><box><1><2><11><22></box>",
        parse_warnings=["测试警告"],
        speed={"inference": 12.5},
        stats={
            "output_tokens": 17,
            "batch_id": 3,
            "batch_size": 8,
            "batch_generation_seconds": 2.5,
            "batch_output_tokens": 101,
        },
    )
    image = {"id": 9, "file_name": "9.jpg"}
    record = locate_val.result_to_record(result, image, {"person": {"id": 1, "name": "person"}})
    assert record["predictions"] == [
        {
            "image_id": 9,
            "category_id": 1,
            "category_name": "person",
            "bbox": [1.0, 2.0, 10.0, 20.0],
            "xyxy": [1.0, 2.0, 11.0, 22.0],
        }
    ]
    assert record["unknown_labels"] == ["human"]
    assert record["output_tokens"] == 17
    assert record["batch_id"] == 3
    assert record["batch_size"] == 8
    assert record["error"] is None


def test_kv_pack_and_unpack_preserve_each_real_row():
    def make_kv(length, value):
        tensor = torch.full((1, 2, length, 3), value, dtype=torch.float32)
        return ((tensor, tensor + 10),)

    rows = [make_kv(2, 1), make_kv(4, 2)]
    packed, valid, lengths, maximum = pack_kv_rows(rows, [0, 1], torch.device("cpu"))
    assert lengths == [2, 4] and maximum == 4
    assert valid.tolist() == [[0, 0, 1, 1], [1, 1, 1, 1]]
    restored = unpack_kv_row(packed, 0, 2, 0, 4, 0)
    assert torch.equal(restored[0][0], rows[0][0][0])
    assert torch.equal(restored[0][1], rows[0][0][1])


def test_kv_length_bucket_preserves_real_cache_rows():
    def make_kv(length, value):
        tensor = torch.full((1, 2, length, 3), value, dtype=torch.float32)
        return ((tensor, tensor + 10),)

    rows = [make_kv(2, 1), make_kv(5, 2)]
    packed, valid, lengths, maximum = pack_kv_rows(rows, [0, 1], torch.device("cpu"), length_multiple=8)
    assert maximum == 8
    assert valid.tolist() == [[0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]]
    for index, row in enumerate(rows):
        restored = unpack_kv_row(packed, index, lengths[index], 0, maximum, 0)
        assert torch.equal(restored[0][0], row[0][0])
        assert torch.equal(restored[0][1], row[0][1])


def test_decode_batch_bucket_only_returns_real_rows():
    seen = {}

    class Decoder:
        def __call__(self, *, input_ids, past_key_values, **kwargs):
            seen["input_shape"] = tuple(input_ids.shape)
            seen["kv_shape"] = tuple(past_key_values[0][0].shape)
            hidden = input_ids.to(torch.float32).unsqueeze(-1).expand(-1, -1, 2)
            return SimpleNamespace(last_hidden_state=hidden, past_key_values=past_key_values)

    language_model = SimpleNamespace(model=Decoder(), lm_head=lambda hidden: hidden)
    key = torch.ones(3, 2, 8, 4)
    output = _forward_decode(
        SimpleNamespace(language_model=language_model),
        input_ids=torch.ones(3, 6, dtype=torch.long),
        attention_mask=torch.ones(3, 14, dtype=torch.long),
        position_ids=torch.arange(6).expand(3, -1),
        past_key_values=((key, key.clone()),),
        logits_count=2,
        batch_bucket=8,
    )
    assert seen == {"input_shape": (8, 6), "kv_shape": (8, 2, 8, 4)}
    assert output.logits.shape == (3, 2, 2)


def test_direct_paged_decoder_skips_model_mask_builder():
    seen_masks = []

    class Layer(torch.nn.Module):
        def forward(self, hidden_states, *, attention_mask, **kwargs):
            seen_masks.append(attention_mask)
            return (hidden_states + 1, kwargs["past_key_value"])

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 4)
            self.layers = torch.nn.ModuleList((Layer(), Layer()))
            self.norm = torch.nn.Identity()

        def get_input_embeddings(self):
            return self.embedding

    decoder = Decoder()
    language_model = SimpleNamespace(model=decoder, lm_head=torch.nn.Identity())
    cache = SimpleNamespace()
    output = _forward_paged_decoder(
        SimpleNamespace(language_model=language_model),
        input_ids=torch.tensor([[1, 2], [3, 4]]),
        position_ids=torch.tensor([[0, 1], [0, 1]]),
        past_key_values=cache,
        logits_count=1,
    )
    assert seen_masks == [None, None]
    assert output.logits.shape == (2, 1, 4)
    assert output.past_key_values is cache


def test_moonvit_batches_equal_grids_and_restores_input_order():
    calls = []

    class Model:
        @staticmethod
        def extract_feature(pixel_values, image_grid_hws):
            calls.append((tuple(pixel_values.shape), image_grid_hws.tolist()))
            lengths = (image_grid_hws[:, 0] * image_grid_hws[:, 1]).tolist()
            return list(pixel_values.split([int(length) for length in lengths], dim=0))

        @staticmethod
        def mlp1(values):
            return values * 2

    inputs = [
        BatchInput(torch.ones(1, 1, dtype=torch.long), torch.full((4, 2), value), torch.tensor([[2, 2]]))
        for value in (1.0, 2.0)
    ]
    inputs.insert(
        1,
        BatchInput(torch.ones(1, 1, dtype=torch.long), torch.full((2, 2), 3.0), torch.tensor([[1, 2]])),
    )
    output = _encode_visual_features(Model(), inputs, torch.float32, visual_batching=True)
    assert len(calls) == 2
    assert calls[0][0] == (8, 2)
    assert [row[0, 0].item() for row in output] == [2.0, 6.0, 4.0]


def test_moonvit_splits_groups_at_npu_tnd_token_limit(monkeypatch):
    calls = []

    class Model:
        @staticmethod
        def extract_feature(pixel_values, image_grid_hws):
            calls.append(int(pixel_values.shape[0]))
            lengths = (image_grid_hws[:, 0] * image_grid_hws[:, 1]).tolist()
            return list(pixel_values.split([int(length) for length in lengths], dim=0))

        @staticmethod
        def mlp1(values):
            return values

    monkeypatch.setattr(locate_batch, "VISUAL_TND_TOKEN_LIMIT", 5)
    inputs = [
        BatchInput(torch.ones(1, 1, dtype=torch.long), torch.full((4, 2), value), torch.tensor([[2, 2]]))
        for value in (1.0, 2.0, 3.0)
    ]
    output = _encode_visual_features(Model(), inputs, torch.float32, visual_batching=True)
    assert calls == [4, 4, 4]
    assert [row[0, 0].item() for row in output] == [1.0, 2.0, 3.0]


def test_expandable_static_kv_cache_reuses_slots_and_grows():
    cache = ExpandableStaticKVCache(slots=2, layers=1, initial_capacity=2)
    positions = torch.tensor([[0, 1], [0, 1]])
    cache.configure_step(torch.tensor([1, 0]), 0, positions, torch.ones_like(positions), required_capacity=2)
    keys = torch.arange(8, dtype=torch.float32).reshape(2, 1, 2, 2)
    values = keys + 100
    attention_keys, attention_values = cache.update(keys, values, 0)
    assert torch.equal(attention_keys, keys)
    assert torch.equal(attention_values, values)

    cache.configure_step(
        torch.tensor([0]),
        2,
        torch.tensor([[2]]),
        torch.tensor([[1]]),
        required_capacity=3,
    )
    new_key = torch.tensor([[[[20.0, 21.0]]]])
    combined, _ = cache.update(new_key, new_key + 100, 0)
    assert cache.capacity >= 3
    assert torch.equal(combined[:, :, :2], keys[1:2])
    assert torch.equal(combined[:, :, 2:], new_key)


def test_paged_kv_cache_preserves_left_padded_view_and_reuses_blocks():
    cache = PagedKVCache(
        slots=2,
        layers=1,
        block_size=2,
        pool_blocks=4,
        max_seq_length=8,
        device=torch.device("cpu"),
    )
    slots = torch.tensor([1, 0])
    positions = torch.tensor([[0, 1], [0, 1]])
    valid = torch.ones_like(positions)
    cache.configure_step(slots, [0, 0], positions, valid, [2, 2], use_paged_attention=False)
    keys = torch.arange(8, dtype=torch.float32).reshape(2, 1, 2, 2)
    returned, _ = cache.update(keys, keys + 100, 0)
    assert torch.equal(returned, keys)

    cache.configure_step(
        torch.tensor([0, 1]),
        [2, 1],
        torch.tensor([[2], [1]]),
        torch.ones(2, 1, dtype=torch.long),
        [3, 2],
        use_paged_attention=False,
    )
    new_keys = torch.tensor([[[[20.0, 21.0]]], [[[30.0, 31.0]]]])
    combined, _ = cache.update(new_keys, new_keys + 100, 0)
    assert combined.shape == (2, 1, 3, 2)
    assert torch.equal(combined[0, :, :2], keys[1])
    assert torch.equal(combined[0, :, 2:], new_keys[0])
    assert torch.equal(combined[1, :, :1], torch.zeros(1, 1, 2))
    assert torch.equal(combined[1, :, 1:2], keys[0, :, :1])
    assert torch.equal(combined[1, :, 2:], new_keys[1])

    before = len(cache._free_blocks)
    cache.release_slot(0)
    assert len(cache._free_blocks) > before


def test_paged_cache_skips_single_token_mask_and_only_uploads_changed_blocks():
    cache = PagedKVCache(
        slots=2,
        layers=1,
        block_size=4,
        pool_blocks=4,
        max_seq_length=16,
        device=torch.device("cpu"),
    )
    slots = torch.tensor([0, 1])
    positions = torch.tensor([[0], [0]])
    valid = torch.ones_like(positions)
    cache.configure_step(slots, [0, 0], positions, valid, [1, 1], use_paged_attention=True)
    version = cache.block_table._version
    assert cache.paged_attention_mask is None
    assert cache._allocated_blocks == [1, 1]
    cache.configure_step(slots, [1, 1], positions, valid, [2, 2], use_paged_attention=True)
    assert cache.block_table._version == version
    cache.configure_step(slots, [3, 3], positions, valid, [5, 5], use_paged_attention=True)
    assert cache.block_table._version > version
    assert cache._allocated_blocks == [2, 2]


def test_paged_kv_cache_import_export_roundtrip():
    cache = PagedKVCache(
        slots=1,
        layers=2,
        block_size=2,
        pool_blocks=4,
        max_seq_length=8,
        device=torch.device("cpu"),
    )
    row_cache = tuple(
        (
            torch.arange(6, dtype=torch.float32).reshape(1, 1, 3, 2) + layer * 20,
            torch.arange(6, dtype=torch.float32).reshape(1, 1, 3, 2) + layer * 20 + 10,
        )
        for layer in range(2)
    )
    assert cache.import_row(0, row_cache) == 3
    restored = cache.export_row(0, 3)
    for expected_layer, restored_layer in zip(row_cache, restored):
        assert torch.equal(expected_layer[0], restored_layer[0])
        assert torch.equal(expected_layer[1], restored_layer[1])


def test_paged_kv_cache_grows_logical_block_table_for_later_long_prompt():
    cache = PagedKVCache(
        slots=1,
        layers=1,
        block_size=2,
        pool_blocks=2,
        max_seq_length=2,
        device=torch.device("cpu"),
    )
    cache.configure_step(
        torch.tensor([0]),
        [0],
        torch.arange(5).view(1, -1),
        torch.ones(1, 5, dtype=torch.long),
        [5],
        use_paged_attention=False,
    )
    assert cache.max_blocks_per_sequence >= 3
    assert cache.block_table.shape[1] == cache.max_blocks_per_sequence


def test_per_image_generators_are_reproducible_and_row_independent():
    first = make_row_generators(torch.device("cpu"), [11, 22])
    second = make_row_generators(torch.device("cpu"), [22, 11])
    first_values = [torch.rand(5, generator=generator) for generator in first]
    second_values = [torch.rand(5, generator=generator) for generator in second]
    assert torch.equal(first_values[0], second_values[1])
    assert torch.equal(first_values[1], second_values[0])


def test_q_sample_buffer_is_reused_and_keeps_independent_rng():
    import ultralytics.models.locateanything.batch as locate_batch

    locate_batch._Q_SAMPLE_BUFFERS.clear()
    logits = torch.zeros(2, 3, 7)
    first_generators = make_row_generators(torch.device("cpu"), [11, 22])
    first = _fill_q_sample_buffer(logits, 3, first_generators, [0, 1]).clone()
    data_ptr = next(iter(locate_batch._Q_SAMPLE_BUFFERS.values())).data_ptr()
    second_generators = make_row_generators(torch.device("cpu"), [11, 22])
    second = _fill_q_sample_buffer(logits, 3, second_generators, [0, 1]).clone()
    assert torch.equal(first, second)
    assert data_ptr == next(iter(locate_batch._Q_SAMPLE_BUFFERS.values())).data_ptr()
    assert not torch.equal(first[:3], first[3:])


def test_continuous_batching_refills_fixed_slots(monkeypatch):
    import ultralytics.models.locateanything.batch as locate_batch

    prefill_sizes = []
    decode_rows = []

    def fake_visual(model, inputs, dtype):
        return [torch.zeros(1, 2) for _ in inputs]

    def fake_prefill(model, prompt_ids, visual_features, image_token_id, pad_token_id, device, **kwargs):
        prefill_sizes.append(len(prompt_ids))
        return [(object(),) for _ in prompt_ids]

    def fake_step_mtp(*args):
        rows, full_ids, generated_ids, finished = args[3], args[5], args[6], args[8]
        decode_rows.append(list(rows))
        for row in rows:
            token = 100 + row
            generated_ids[row].append(token)
            full_ids[row].append(token)
            finished[row] = True

    monkeypatch.setattr(locate_batch, "_encode_visual_features", fake_visual)
    monkeypatch.setattr(locate_batch, "_prefill_prompt_rows", fake_prefill)
    monkeypatch.setattr(locate_batch, "_step_mtp", fake_step_mtp)
    model = SimpleNamespace(
        config=SimpleNamespace(image_token_index=9),
        token_ids={"im_end_token_id": 2, "default_mask_token_id": 3},
    )
    tokenizer = SimpleNamespace(
        pad_token_id=0,
        model_max_length=100,
        decode=lambda ids, **kwargs: ",".join(str(value) for value in ids),
    )
    inputs = [BatchInput(torch.tensor([[index + 1]]), torch.zeros(1), None) for index in range(5)]
    outputs = generate_batch_hybrid(
        model,
        tokenizer,
        inputs,
        device=torch.device("cpu"),
        dtype=torch.float32,
        seeds=list(range(5)),
        max_new_tokens=4,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        slot_capacity=2,
    )
    assert prefill_sizes == [2, 2, 1]
    assert decode_rows == [[0, 1], [2, 3], [4]]
    assert [output.text for output in outputs] == [str(100 + index) for index in range(5)]


def test_streaming_provider_refills_at_low_watermark(monkeypatch):
    import ultralytics.models.locateanything.batch as locate_batch

    prefill_sizes, provider_requests, completed_rows = [], [], []
    pending = [BatchInput(torch.tensor([[index + 1]]), torch.zeros(1), None) for index in range(6)]

    def provider(count):
        provider_requests.append(count)
        items = pending[:count]
        del pending[:count]
        return items, list(range(6 - len(pending) - len(items), 6 - len(pending)))

    def fake_visual(model, inputs, dtype):
        return [torch.zeros(1, 2) for _ in inputs]

    def fake_prefill(model, prompt_ids, visual_features, image_token_id, pad_token_id, device, **kwargs):
        prefill_sizes.append(len(prompt_ids))
        return [(object(),) for _ in prompt_ids]

    def fake_step_mtp(*args):
        rows, full_ids, generated_ids, finished = args[3], args[5], args[6], args[8]
        row = rows[0]
        generated_ids[row].append(100 + row)
        full_ids[row].append(100 + row)
        finished[row] = True

    monkeypatch.setattr(locate_batch, "_encode_visual_features", fake_visual)
    monkeypatch.setattr(locate_batch, "_prefill_prompt_rows", fake_prefill)
    monkeypatch.setattr(locate_batch, "_step_mtp", fake_step_mtp)
    model = SimpleNamespace(
        config=SimpleNamespace(image_token_index=9),
        token_ids={"im_end_token_id": 2, "default_mask_token_id": 3},
    )
    tokenizer = SimpleNamespace(
        pad_token_id=0,
        model_max_length=100,
        decode=lambda ids, **kwargs: ",".join(str(value) for value in ids),
    )
    outputs = generate_batch_hybrid(
        model,
        tokenizer,
        [],
        device=torch.device("cpu"),
        dtype=torch.float32,
        seeds=[],
        max_new_tokens=4,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        slot_capacity=4,
        refill_batch_size=2,
        input_provider=provider,
        completion_callback=lambda rows: completed_rows.extend(row for row, _ in rows),
        max_provider_inputs=6,
    )
    assert prefill_sizes == [4, 2]
    assert provider_requests == [4, 2, 2]
    assert completed_rows == list(range(6))
    assert [output.text for output in outputs] == [str(100 + index) for index in range(6)]


def test_vectorized_repetition_penalty_matches_token_semantics():
    logits = torch.tensor(
        [
            [[2.0, -3.0, 4.0, 5.0], [1.0, -2.0, -4.0, 6.0]],
            [[-2.0, 3.0, -4.0, 5.0], [-1.0, 2.0, 4.0, -6.0]],
        ]
    )
    generated = torch.tensor([[1, 2, 2], [0, 3, 3]])
    actual = _apply_repetition_penalty(logits.clone(), generated, 2.0)
    expected = logits.clone()
    expected[0, :, 1] *= 2.0
    expected[0, 0, 2] /= 2.0
    expected[0, 1, 2] *= 2.0
    expected[1, :, 0] *= 2.0
    expected[1, 0, 3] /= 2.0
    expected[1, 1, 3] *= 2.0
    assert torch.equal(actual, expected)


def test_device_token_history_matches_host_repetition_and_reuses_slots():
    history = DeviceTokenHistory(2, 2, 9, torch.device("cpu"))
    rows = [[1, 2, 3], [2, 4]]
    history.reset_slots([1, 0], rows)
    history.append([1, 0], [[5], [6, 7]])
    logits = torch.randn(2, 3, 10)
    device_tokens = history.select([1, 0])
    host_tokens = torch.tensor([[1, 2, 3, 5], [2, 4, 6, 7]])
    assert torch.equal(
        _apply_repetition_penalty(logits.clone(), device_tokens, 1.1),
        _apply_repetition_penalty(logits.clone(), host_tokens, 1.1),
    )
    history.release_slot(1)
    history.reset_slots([1], [[8]])
    assert history.select([1]).tolist() == [[8]]


def test_qsample_reservoir_is_reproducible_and_reuses_prefetched_positions():
    first = QSampleReservoir(2, 12, torch.device("cpu"))
    second = QSampleReservoir(2, 12, torch.device("cpu"))
    generators_a = [torch.Generator().manual_seed(11), torch.Generator().manual_seed(29)]
    generators_b = [torch.Generator().manual_seed(11), torch.Generator().manual_seed(29)]

    first_chunk = first.take(6, [0, 1], generators_a, [0, 1], 7)
    second_chunk = first.take(6, [0, 1], generators_a, [0, 1], 7)
    repeated_first = second.take(6, [0, 1], generators_b, [0, 1], 7)
    repeated_second = second.take(6, [0, 1], generators_b, [0, 1], 7)

    assert first_chunk.shape == (12, 7)
    assert first.cursors == [12, 12]
    assert torch.equal(first_chunk, repeated_first)
    assert torch.equal(second_chunk, repeated_second)
    assert not torch.equal(first_chunk[:6], first_chunk[6:])
    assert first.take(1, [0], generators_a, [0], 7) is None
    first.release_slot(1)
    assert first.cursors[1] == 12


def test_paged_cache_reuses_step_write_metadata_across_layers():
    cache = PagedKVCache(
        slots=2,
        layers=2,
        block_size=4,
        pool_blocks=4,
        max_seq_length=8,
        device=torch.device("cpu"),
    )
    slots = torch.tensor([0, 1])
    positions = torch.tensor([[0, 1], [0, 1]])
    valid = torch.ones_like(positions)
    cache.configure_step(slots, [0, 0], positions, valid, [2, 2], use_paged_attention=False)
    local_rows = cache._write_local_rows
    physical_blocks = cache._write_physical_blocks
    key = torch.arange(16, dtype=torch.float32).reshape(2, 1, 2, 4)
    cache.update(key, key + 100, 0)
    cache.update(key + 200, key + 300, 1)
    assert cache._write_local_rows is local_rows
    assert cache._write_physical_blocks is physical_blocks
    assert torch.equal(cache.key_cache[0][physical_blocks, :, cache._write_offsets], key.reshape(4, 1, 4))
    assert torch.equal(cache.key_cache[1][physical_blocks, :, cache._write_offsets], (key + 200).reshape(4, 1, 4))


def test_paged_sparse_causal_mask_and_sampling_constants_are_reused():
    device = torch.device("cpu")
    first_mask = _paged_right_down_causal_mask(device)
    second_mask = _paged_right_down_causal_mask(device)
    assert first_mask.data_ptr() == second_mask.data_ptr()
    assert first_mask.shape == (2048, 2048)
    assert first_mask.dtype == torch.bool
    assert first_mask[0, 1] and not first_mask[1, 0]

    first_k, first_p = _sample_parameter_tensors(12, 1024, 0.9, torch.bfloat16, device)
    second_k, second_p = _sample_parameter_tensors(12, 1024, 0.9, torch.bfloat16, device)
    assert first_k.data_ptr() == second_k.data_ptr()
    assert first_p.data_ptr() == second_p.data_ptr()
    assert first_k.tolist() == [1024] * 12
    assert torch.allclose(first_p.float(), torch.full((12,), 0.9), atol=0.01)


def test_mixed_paged_step_advances_ar_and_mtp_in_one_forward(monkeypatch):
    import ultralytics.models.locateanything.batch as locate_batch

    seen = {}

    def fake_forward(model, **kwargs):
        seen["input_ids"] = kwargs["input_ids"].clone()
        seen["position_ids"] = kwargs["position_ids"].clone()
        return SimpleNamespace(logits=torch.zeros(2, 6, 32), past_key_values=kwargs["past_key_values"])

    def fake_probabilities(model, logits, *args, **kwargs):
        assert logits.shape == (1, 1, 32)
        return torch.tensor([[model.token_ids["box_end_token_id"]]]), None

    def fake_mtp(model, logits, *args, **kwargs):
        assert logits.shape == (1, 6, 32)
        return [{"type": "error_box", "tokens": [model.token_ids["box_start_token_id"]]}]

    monkeypatch.setattr(locate_batch, "_forward_decode", fake_forward)
    monkeypatch.setattr(locate_batch, "_sample_probabilities", fake_probabilities)
    monkeypatch.setattr(locate_batch, "_sample_mtp", fake_mtp)
    token_ids = {
        "box_start_token_id": 10,
        "box_end_token_id": 11,
        "none_token_id": 12,
        "coord_start_token_id": 20,
        "coord_end_token_id": 25,
        "im_end_token_id": 2,
    }
    model = SimpleNamespace(token_ids=token_ids)
    cache = PagedKVCache(
        slots=2,
        layers=1,
        block_size=4,
        pool_blocks=4,
        max_seq_length=16,
        device=torch.device("cpu"),
    )
    cached_lengths = [1, 2]
    full_ids = [[4, 5], [6, 7]]
    generated_ids = [[], []]
    modes = ["ar", "mtp"]
    finished = [False, False]
    _step_mixed_paged(
        model,
        [0],
        [1],
        cached_lengths,
        full_ids,
        generated_ids,
        modes,
        finished,
        [20, 20],
        0,
        3,
        9,
        0.0,
        1.0,
        1.0,
        make_row_generators(torch.device("cpu"), [1, 2]),
        torch.device("cpu"),
        cache,
        [0, 1],
    )
    assert seen["input_ids"].tolist() == [[0, 0, 0, 0, 0, 5], [7, 3, 3, 3, 3, 3]]
    assert seen["position_ids"].tolist() == [[1, 1, 1, 1, 1, 1], [1, 2, 3, 4, 5, 6]]
    assert cached_lengths == [2, 2]
    assert generated_ids == [[11], [10]]
    assert full_ids == [[4, 5, 11], [6, 7, 10]]
    assert modes == ["mtp", "ar"]


def test_vectorized_mtp_decode_and_python_pattern_handler():
    token_ids = {
        "box_start_token_id": 10,
        "box_end_token_id": 11,
        "ref_start_token_id": 12,
        "ref_end_token_id": 13,
        "none_token_id": 14,
        "null_token_id": 15,
        "im_end_token_id": 16,
        "coord_start_token_id": 20,
        "coord_end_token_id": 1020,
    }
    probabilities = torch.full((4, 6, 1030), 1e-8)
    sampled = torch.tensor([[16] * 6, [1] * 6, [2] * 6, [10, 21, 2, 3, 4, 5]])
    probabilities[0, 0, 10] = 0.8
    probabilities[0, 1, 14] = 0.3
    probabilities[0, 2, 11] = 0.3
    probabilities[0, 3, 15] = probabilities[0, 4, 15] = 0.2
    probabilities[1, 5, 11] = 0.3
    for position, token in enumerate((21, 22, 23, 24), 1):
        probabilities[1, position, token] = 0.95
    probabilities[2, 0, 12] = 0.7
    for position, token in enumerate((1, 2, 3, 4, 13), 1):
        probabilities[2, position, token] = 0.9
    probabilities /= probabilities.sum(dim=-1, keepdim=True)

    decoded = _decode_mtp_tokens(sampled, probabilities, token_ids)
    patterns = [_handle_pattern_tokens(tokens, token_ids) for tokens in decoded]
    assert decoded[:3] == [
        [10, 14, 11, 15, 15, 15],
        [10, 21, 22, 23, 24, 11],
        [12, 1, 2, 3, 4, 13],
    ]
    assert [pattern["type"] for pattern in patterns] == ["empty_box", "coord_box", "ref_object", "error_box"]
    assert patterns[3]["tokens"] == [10, 21]


def test_duplicate_box_guard_stops_only_consecutive_identical_patterns():
    last = [None]
    counts = [0]
    box = {"type": "coord_box", "tokens": [10, 21, 22, 23, 24, 11]}
    for _ in range(3):
        guarded, stopped = _guard_duplicate_box_pattern(box, 0, last, counts, 3, 16)
        assert guarded == box and not stopped
    guarded, stopped = _guard_duplicate_box_pattern(box, 0, last, counts, 3, 16)
    assert guarded == {"type": "im_end", "tokens": [16]} and stopped

    ref = {"type": "ref_object", "tokens": [12, 1, 13]}
    _guard_duplicate_box_pattern(ref, 0, last, counts, 3, 16)
    guarded, stopped = _guard_duplicate_box_pattern(box, 0, last, counts, 3, 16)
    assert guarded == box and not stopped and counts[0] == 1


def test_compact_mtp_probabilities_match_full_distribution():
    token_ids = {
        "box_start_token_id": 10,
        "box_end_token_id": 11,
        "ref_start_token_id": 12,
        "ref_end_token_id": 13,
        "none_token_id": 14,
        "null_token_id": 15,
        "im_end_token_id": 16,
        "coord_start_token_id": 20,
        "coord_end_token_id": 1020,
    }
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn(4, 6, 1030, generator=generator)
    probabilities = torch.softmax(logits, dim=-1)
    sampled = probabilities.argmax(dim=-1)
    full = _decode_mtp_tokens(sampled, probabilities, token_ids)
    compact_from_probabilities = _summarize_mtp_distribution(probabilities, token_ids, input_is_logits=False)
    compact_from_logits = _summarize_mtp_distribution(logits, token_ids, input_is_logits=True)
    assert _decode_mtp_tokens(sampled, compact_from_probabilities, token_ids) == full
    assert _decode_mtp_tokens(sampled, compact_from_logits, token_ids) == full


def test_candidate_top_p_coverage_matches_full_softmax():
    logits = torch.randn(3, 6, 30, generator=torch.Generator().manual_seed(19)) * 1.7
    values, ids, keep, coverage = _candidate_top_p_values(logits, 0.9, candidate_size=24)
    assert bool((coverage > 0.9).all())
    full_probabilities = torch.softmax(logits, dim=-1)
    expected_coverage = full_probabilities.gather(-1, ids).sum(dim=-1)
    assert torch.allclose(coverage, expected_coverage, atol=1e-6, rtol=1e-6)
    candidate_probabilities = full_probabilities.gather(-1, ids)
    expected_keep = (candidate_probabilities.cumsum(-1) - candidate_probabilities) <= 0.9
    assert torch.equal(keep, expected_keep)


def test_candidate_top_p_detects_insufficient_coverage_for_fallback():
    logits = torch.zeros(1, 6, 100)
    _, _, _, coverage = _candidate_top_p_values(logits, 0.9, candidate_size=10)
    assert torch.allclose(coverage, torch.full_like(coverage, 0.1))


def test_probability_none_mode_discards_distribution_but_keeps_sample():
    logits = torch.tensor([[[1.0, 4.0, 2.0]]])
    sampled, probabilities = _sample_probabilities(
        SimpleNamespace(),
        logits,
        temperature=0.0,
        top_p=1.0,
        generators=[],
        global_rows=[0],
        probability_mode="none",
    )
    assert sampled.tolist() == [[1]]
    assert probabilities is None


def test_f1_one_to_one_matching_and_crowd_ignore():
    records = [
        _record(
            1,
            [
                _prediction(1, [0.0, 0.0, 10.0, 10.0]),
                _prediction(1, [0.0, 0.0, 10.0, 10.0]),
                _prediction(1, [20.0, 0.0, 30.0, 10.0]),
            ],
        )
    ]
    annotations = [
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "iscrowd": 0},
        {"image_id": 1, "category_id": 1, "bbox": [20, 0, 10, 10], "iscrowd": 1},
    ]
    metrics = locate_val.compute_locate_metrics(
        records,
        annotations,
        [{"id": 1, "name": "person"}],
        {1},
    )
    for threshold in ("0.50", "0.95"):
        micro = metrics["f1"][threshold]["micro"]
        assert micro["tp"] == 1
        assert micro["fp"] == 1
        assert micro["fn"] == 0
        assert micro["f1"] == pytest.approx(2 / 3)
    assert metrics["mean_gt_iou"] == pytest.approx(1.0)
    assert metrics["evaluated_non_crowd_gt"] == 1


def test_paper_matching_uses_generation_order_instead_of_global_iou(monkeypatch):
    overlaps = {(1, 10): 0.8, (1, 20): 0.0, (2, 10): 0.9, (2, 20): 0.8}
    monkeypatch.setattr(locate_val, "bbox_iou", lambda prediction, target: overlaps[prediction[0], target[0]])
    counts, matched = locate_val._paper_match_counts([[1], [2]], [[10], [20]], [], 0.5)
    assert counts == {"tp": 2, "fp": 0, "fn": 0}
    assert matched == [0.8, 0.8]


def test_paper_matching_limits_each_image_category_to_first_100_predictions():
    predictions = [[0.0, 0.0, 10.0, 10.0] for _ in range(101)]
    counts, _ = locate_val._paper_match_counts(predictions, [[0.0, 0.0, 10.0, 10.0]], [], 0.5)
    assert counts == {"tp": 1, "fp": 99, "fn": 0}


def test_paper_metrics_use_positive_only_ten_thresholds_and_harmonic_macro():
    records = [
        _record(
            1,
            [
                _prediction(1, [0.0, 0.0, 10.0, 10.0], 1),
                _prediction(1, [20.0, 20.0, 30.0, 30.0], 2),
            ],
        ),
        _record(
            2,
            [
                _prediction(2, [20.0, 20.0, 30.0, 30.0], 2),
                _prediction(2, [0.0, 0.0, 10.0, 10.0], 2),
            ],
        ),
        _record(3, [_prediction(3, [40.0, 40.0, 50.0, 50.0], 1)]),
    ]
    annotations = [
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "iscrowd": 0},
        {"image_id": 2, "category_id": 2, "bbox": [0, 0, 10, 10], "iscrowd": 0},
    ]
    metrics = locate_val.compute_locate_metrics(
        records,
        annotations,
        [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}],
        {1, 2, 3},
    )
    assert list(metrics["f1"]) == [f"{value / 100:.2f}" for value in range(50, 100, 5)] + ["mean"]
    assert metrics["positive_only_dropped_predictions"] == 1
    assert metrics["f1"]["0.50"]["per_class"]["person"]["count_precision"] == pytest.approx(0.5)
    at_50 = metrics["f1"]["0.50"]["macro"]
    assert at_50["precision"] == pytest.approx(0.75)
    assert at_50["recall"] == pytest.approx(1.0)
    assert at_50["f1"] == pytest.approx(6 / 7)
    assert metrics["f1"]["mean"]["macro"]["f1"] == pytest.approx(6 / 7)


def test_resume_rejects_legacy_shards_in_paper_protocol(tmp_path):
    record = _record(1, [])
    (tmp_path / "predictions.rank0.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="旧版/未记录"):
        locate_val.validate_resume_protocol(tmp_path, 1, "paper")
    locate_val.validate_resume_protocol(tmp_path, 1, "legacy")
    record["validation_preprocess"] = {"protocol_id": PAPER_PROTOCOL_ID}
    (tmp_path / "predictions.rank0.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    locate_val.validate_resume_protocol(tmp_path, 1, "paper")


def test_constant_score_predictions_are_explicitly_non_native():
    records = [_record(3, [_prediction(3, [1.0, 2.0, 6.0, 10.0])])]
    predictions = locate_val.build_constant_score_predictions(records)
    assert predictions == [{"image_id": 3, "category_id": 1, "bbox": [1.0, 2.0, 5.0, 8.0], "score": 1.0}]


def test_jsonl_merge_tolerates_partial_tail_and_uses_latest_record(tmp_path):
    first = _record(2, [])
    latest = _record(2, [_prediction(2, [0.0, 0.0, 2.0, 2.0])])
    rank0 = tmp_path / "predictions.rank0.jsonl"
    rank0.write_text("\n".join(json.dumps(item) for item in (first, latest)), encoding="utf-8")
    rank1 = tmp_path / "predictions.rank1.jsonl"
    rank1.write_text(json.dumps(_record(1, [])) + "\n{broken", encoding="utf-8")
    records = locate_val.merge_prediction_shards(tmp_path, 2)
    assert [record["image_id"] for record in records] == [1, 2]
    assert len(records[1]["predictions"]) == 1


def test_jsonl_merge_prefers_success_over_error_from_another_rank(tmp_path):
    success = _record(1, [_prediction(1, [0.0, 0.0, 2.0, 2.0])])
    failed = _record(1, [])
    failed["error"] = "旧错误"
    (tmp_path / "predictions.rank0.jsonl").write_text(json.dumps(success) + "\n", encoding="utf-8")
    (tmp_path / "predictions.rank1.jsonl").write_text(json.dumps(failed) + "\n", encoding="utf-8")
    records = locate_val.merge_prediction_shards(tmp_path, 2)
    assert records == [success]


def test_parent_progress_incrementally_deduplicates_resume_records(tmp_path, monkeypatch):
    class DummyProgress:
        def __init__(self, *, initial=0, **kwargs):
            self.n = initial

        def update(self, value):
            self.n += value

        def set_postfix(self, **kwargs):
            self.postfix = kwargs

        def close(self):
            pass

    monkeypatch.setattr(locate_val, "TQDM", lambda **kwargs: DummyProgress(**kwargs))
    first = _record(1, [])
    rank0 = tmp_path / "predictions.rank0.jsonl"
    rank0.write_text(json.dumps(first) + "\n", encoding="utf-8")
    monitor = locate_val._DistributedProgress(tmp_path, world_size=2, total=2)
    duplicate = _record(1, [])
    duplicate.update(batch_id=10, batch_output_tokens=10, batch_generation_seconds=2.0)
    second = _record(2, [])
    second.update(batch_id=20, batch_output_tokens=20, batch_generation_seconds=2.0)
    with rank0.open("a", encoding="utf-8") as file:
        file.write(json.dumps(duplicate) + "\n" + json.dumps(second) + "\n")
    monitor.poll()
    assert monitor.completed == {1, 2}
    assert monitor.progress.n == 2
    assert monitor.tokens_per_second == pytest.approx(7.5)
    monitor.close()


def test_parent_progress_uses_streaming_cumulative_counters(tmp_path, monkeypatch):
    class DummyProgress:
        def __init__(self, **kwargs):
            pass

        def update(self, value):
            pass

        def set_postfix(self, **kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(locate_val, "TQDM", lambda **kwargs: DummyProgress(**kwargs))
    monitor = locate_val._DistributedProgress(tmp_path, world_size=2, total=2)
    for rank, tokens, seconds in ((0, 30, 3.0), (1, 50, 4.0)):
        record = _record(rank, [])
        record["generation_stats"] = {
            "scheduler_output_tokens": tokens,
            "scheduler_generation_seconds": seconds,
        }
        (tmp_path / f"predictions.rank{rank}.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    monitor.poll()
    assert monitor.tokens_per_second == pytest.approx(20.0)
    monitor.close()


def test_validator_serializes_distributed_configuration(tmp_path):
    owner = SimpleNamespace(
        model_name="nvidia/LocateAnything-3B",
        revision="fixed-revision",
        device=torch.device("cpu"),
        model=torch.nn.Linear(2, 2),
    )
    validator = locate_val.LocateAnythingValidator(
        model=owner,
        data="coco.yaml",
        device=locate_val.DEFAULT_DEVICES,
        output_dir=tmp_path,
        max_images=8,
        callbacks_=defaultdict(list),
    )
    config = validator._serializable_config()
    assert config["model"] == owner.model_name
    assert config["revision"] == owner.revision
    assert config["devices"] == locate_val.DEFAULT_DEVICES
    assert config["max_images"] == 8
    assert config["global_batch"] == 1
    assert config["batch"] == 1
    assert config["scheduler"] == "pipeline"
    assert config["protocol"] == "paper"
    assert config["continuous_window"] == 1
    assert config["continuous_batching"] is False
    assert config["dynamic_scheduling"] is False
    assert config["refill_batch"] == 0
    assert config["static_kv_cache"] is False
    assert config["paged_kv_cache"] is False
    assert config["max_duplicate_boxes"] == 0
    assert config["shape_bucketing"] is False
    assert config["kv_bucket_size"] == 128
    assert config["npu_graph"] is False
    assert config["visual_batching"] is False
    assert config["direct_paged_decode"] is True
    assert config["device_repetition_cache"] is True
    assert config["qsample_reservoir"] is False
    assert config["overlap_prefill"] is True
    assert config["candidate_top_p"] is True
    assert config["cpu_affinity"] is True
    assert config["npu_fast_path"] == "auto"
    assert config["fused_qkv"] is False
    assert config["fused_add_rms_norm"] is True
    assert config["fused_mlp"] is False


@pytest.mark.parametrize("batch", [2, 3, 7, 8, 16])
def test_validator_accepts_arbitrary_positive_batch(tmp_path, batch):
    owner = SimpleNamespace(
        model_name="nvidia/LocateAnything-3B",
        revision="fixed-revision",
        device=torch.device("cpu"),
        model=torch.nn.Linear(2, 2),
    )
    validator = locate_val.LocateAnythingValidator(
        model=owner,
        device=locate_val.DEFAULT_DEVICES,
        output_dir=tmp_path,
        batch=batch,
        callbacks_=defaultdict(list),
    )
    assert validator.args.batch == batch
    assert validator.args.global_batch == batch
    assert validator.args.continuous_batching is True
    assert validator.args.dynamic_scheduling is False  # 单卡无需跨rank调度
    assert validator.args.refill_batch == min(8, batch)
    assert validator.args.paged_kv_cache is True
    assert validator.args.visual_batching is True


def test_validator_splits_strict_global_batch_across_devices(tmp_path):
    owner = SimpleNamespace(
        model_name="nvidia/LocateAnything-3B",
        revision="fixed-revision",
        device=torch.device("cpu"),
        model=torch.nn.Linear(2, 2),
    )
    validator = locate_val.LocateAnythingValidator(
        model=owner,
        device="0,1,2,3",
        output_dir=tmp_path,
        batch=32,
        callbacks_=defaultdict(list),
    )
    assert validator.world_size == 4
    assert validator.args.global_batch == 32
    assert validator.args.batch == 8
    assert validator.args.continuous_batching is True
    assert validator.args.dynamic_scheduling is True

    with pytest.raises(ValueError, match="全局batch"):
        locate_val.LocateAnythingValidator(
            model=owner,
            device="0,1,2,3",
            output_dir=tmp_path,
            batch=10,
            callbacks_=defaultdict(list),
        )


def test_validator_manual_torchrun_uses_all_local_devices_when_device_is_none(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.7")
    monkeypatch.setenv("MASTER_PORT", "23100")
    owner = SimpleNamespace(
        model_name="nvidia/LocateAnything-3B",
        revision="fixed-revision",
        device=torch.device("cpu"),
        model=torch.nn.Linear(2, 2),
    )
    validator = locate_val.LocateAnythingValidator(
        model=owner,
        device=None,
        output_dir=tmp_path,
        batch=16,
        callbacks_=defaultdict(list),
    )
    assert validator.device_ids == [0, 1, 2, 3]
    assert validator.local_world_size == 4
    assert validator.world_size == 8
    assert validator.args.batch == 2
    assert validator.args.store_host == "10.0.0.7"
    assert validator.args.store_port == 23101
    assert validator.args.nnodes == 2


def test_validator_resume_rejects_incompatible_distributed_layout(tmp_path):
    owner = SimpleNamespace(
        model_name="nvidia/LocateAnything-3B",
        revision="fixed-revision",
        device=torch.device("cpu"),
        model=torch.nn.Linear(2, 2),
    )
    initial = locate_val.LocateAnythingValidator(
        model=owner,
        device="0,1",
        output_dir=tmp_path,
        batch=8,
        callbacks_=defaultdict(list),
    )
    initial._prepare_run_config(tmp_path)

    resumed = locate_val.LocateAnythingValidator(
        model=owner,
        device="0,1,2,3",
        output_dir=tmp_path,
        batch=8,
        resume=True,
        callbacks_=defaultdict(list),
    )
    with pytest.raises(RuntimeError, match="布局不一致"):
        resumed._prepare_run_config(tmp_path)


def test_validator_allows_explicitly_disabling_adaptive_batch_defaults(tmp_path):
    owner = SimpleNamespace(
        model_name="nvidia/LocateAnything-3B",
        revision="fixed-revision",
        device=torch.device("cpu"),
        model=torch.nn.Linear(2, 2),
    )
    validator = locate_val.LocateAnythingValidator(
        model=owner,
        device=locate_val.DEFAULT_DEVICES,
        output_dir=tmp_path,
        batch=128,
        continuous_batching=False,
        dynamic_scheduling=False,
        refill_batch=0,
        paged_kv_cache=False,
        visual_batching=False,
        callbacks_=defaultdict(list),
    )
    assert validator.args.continuous_batching is False
    assert validator.args.dynamic_scheduling is False
    assert validator.args.refill_batch == 0
    assert validator.args.paged_kv_cache is False
    assert validator.args.visual_batching is False


@pytest.mark.parametrize("batch", [0, -1, 1.5, True, "8"])
def test_validator_rejects_nonpositive_or_noninteger_batch(tmp_path, batch):
    owner = SimpleNamespace(model_name="m", revision="r", device=torch.device("cpu"), model=torch.nn.Linear(1, 1))
    with pytest.raises(ValueError, match="batch"):
        locate_val.LocateAnythingValidator(
            model=owner,
            device=locate_val.DEFAULT_DEVICES,
            output_dir=tmp_path,
            batch=batch,
            callbacks_=defaultdict(list),
        )


def test_batch_greater_than_one_requires_hybrid(tmp_path):
    owner = SimpleNamespace(model_name="m", revision="r", device=torch.device("cpu"), model=torch.nn.Linear(1, 1))
    with pytest.raises(ValueError, match="generation_mode"):
        locate_val.LocateAnythingValidator(
            model=owner,
            device=locate_val.DEFAULT_DEVICES,
            output_dir=tmp_path,
            batch=8,
            generation_mode="slow",
            callbacks_=defaultdict(list),
        )


def test_validator_accepts_combined_continuous_dynamic_mode(tmp_path):
    owner = SimpleNamespace(model_name="m", revision="r", device=torch.device("cpu"), model=torch.nn.Linear(1, 1))
    validator = locate_val.LocateAnythingValidator(
        model=owner,
        device=locate_val.DEFAULT_DEVICES,
        output_dir=tmp_path,
        batch=128,
        continuous_batching=True,
        dynamic_scheduling=True,
        refill_batch=32,
        callbacks_=defaultdict(list),
    )
    assert validator.args.continuous_batching is True
    assert validator.args.dynamic_scheduling is True
    assert validator.args.refill_batch == 32


def test_validator_accepts_continuous_paged_and_visual_batching(tmp_path):
    owner = SimpleNamespace(model_name="m", revision="r", device=torch.device("cpu"), model=torch.nn.Linear(1, 1))
    validator = locate_val.LocateAnythingValidator(
        model=owner,
        device=locate_val.DEFAULT_DEVICES,
        output_dir=tmp_path,
        batch=64,
        continuous_batching=True,
        paged_kv_cache=True,
        visual_batching=True,
        callbacks_=defaultdict(list),
    )
    assert validator.args.paged_kv_cache is True
    assert validator.args.visual_batching is True


def test_graph_worker_environment_is_local(monkeypatch):
    monkeypatch.setenv("TASK_QUEUE_ENABLE", "2")
    assert locate_val._distributed_worker_env(False)["TASK_QUEUE_ENABLE"] == "2"
    assert locate_val._distributed_worker_env(True)["TASK_QUEUE_ENABLE"] == "1"
    assert locate_val.os.environ["TASK_QUEUE_ENABLE"] == "2"


def test_model_val_returns_and_stores_validator_metrics(monkeypatch):
    expected = object()
    seen = {}

    class FakeValidator:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def __call__(self):
            return expected

    monkeypatch.setattr(locate_val, "LocateAnythingValidator", FakeValidator)
    owner = object.__new__(LocateAnything)
    owner.callbacks = defaultdict(list)
    owner.metrics = None
    result = LocateAnything.val(owner, data="coco.yaml", device=locate_val.DEFAULT_DEVICES, batch=8, max_images=8)
    assert result is expected and owner.metrics is expected
    assert seen["model"] is owner
    assert seen["data"] == "coco.yaml"
    assert seen["device"] == locate_val.DEFAULT_DEVICES
    assert seen["batch"] == 8
    assert seen["scheduler"] == "pipeline"
    assert seen["protocol"] == "paper"
    assert seen["max_images"] == 8


def test_global_token_speed_and_average_tokens():
    records = [_record(1, []), _record(2, [])]
    records[0]["output_tokens"] = 8
    records[1]["output_tokens"] = 12
    speed = locate_val._aggregate_speed(
        records,
        global_wall_seconds=4.0,
        processed=2,
        processed_boxes=0,
        processed_tokens=80,
        global_generation_seconds=2.0,
    )
    assert speed["output_tokens"] == 20
    assert speed["average_tokens_per_image"] == 10
    assert speed["tokens_per_second"] == 40


def test_finalize_writes_all_public_artifacts(tmp_path, monkeypatch):
    record = _record(1, [_prediction(1, [0.0, 0.0, 10.0, 10.0])])
    (tmp_path / "predictions.rank0.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        locate_val,
        "run_nonstandard_coco_ap",
        lambda *args, **kwargs: {
            "status": "ok_nonstandard_constant_score",
            "score_policy": "constant_1.0",
            "warning": "固定分数",
            "AP50_95": 0.1,
            "AP50": 0.2,
            "AP75": 0.05,
        },
    )
    args = Namespace(
        model="nvidia/LocateAnything-3B",
        data="coco.yaml",
        devices=locate_val.DEFAULT_DEVICES,
        generation_mode="hybrid",
        max_new_tokens=8192,
        temperature=0.7,
        top_p=0.9,
        seed=0,
        max_images=1,
    )
    coco = {
        "annotation_path": str(tmp_path / "instances.json"),
        "images": [{"id": 1}],
        "annotations": [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "iscrowd": 0}],
        "categories": [{"id": 1, "name": "person"}],
    }
    metrics = locate_val.finalize_results(
        args,
        tmp_path,
        coco,
        1,
        global_wall_seconds=2.0,
        processed=1,
        processed_boxes=1,
    )
    for name in ("predictions.json", "metrics.json", "summary.txt"):
        assert (tmp_path / name).is_file()
    assert metrics["official_locate_metrics"]["f1"]["0.50"]["micro"]["f1"] == 1.0
    assert metrics["nonstandard_constant_score_coco_ap"]["score_policy"] == "constant_1.0"
    assert json.loads((tmp_path / "predictions.json").read_text())[0]["score"] == 1.0
    assert "不是标准COCO AP" in (tmp_path / "summary.txt").read_text(encoding="utf-8")
    metrics_object = locate_val.LocateMetrics(metrics, tmp_path)
    assert metrics_object.save_dir == tmp_path
    assert metrics_object.results_dict["metrics/F1-50(B)"] == 1.0
    assert metrics_object.results_dict["metrics/F1-mean(B)"] == 1.0
    assert metrics_object.fitness == 1.0
    assert metrics_object.summary()[0]["Class"] == "person"
