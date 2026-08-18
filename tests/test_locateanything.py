from __future__ import annotations

import inspect
import json
import subprocess
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ultralytics.engine.runtime import CallbackHost
from ultralytics.models.locateanything.compat import (
    _TorchTensorCopyCompat,
    _moonvit_rope_precompute_compat,
    _patch_processor_class,
    _prepare_4d_causal_attention_mask_compat,
    _repair_qwen_rope_buffers,
    patch_transformers_514,
)
from ultralytics.models.locateanything.data import ConversationDataset, build_pbd_labels, format_yolo_conversation
from ultralytics.models.locateanything.model import _build_prompt
from ultralytics.models.locateanything.results import LocateAnythingResult, parse_locate_output
from ultralytics.models.locateanything.train import (
    LocateAnythingTrainer,
    _clone_embedding_output,
    _disable_remote_auxiliary_training_return,
)


def test_top_level_locateanything_is_lazy():
    code = (
        "import sys; import ultralytics; assert 'transformers' not in sys.modules; "
        "from ultralytics import LocateAnything; assert 'transformers' not in sys.modules; "
        "assert LocateAnything.__name__ == 'LocateAnything'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_locateanything_task_prompts():
    prompt, label = _build_prompt("detect", ["person", "car"], multiple=True, output="box")
    assert prompt == "Locate all the instances that matches the following description: person</c>car."
    assert label == "person</c>car"
    assert _build_prompt("ground", "red car", multiple=False, output="box")[0].startswith("Locate a single instance")
    assert _build_prompt("gui", "search", multiple=True, output="point")[0] == "Point to: search."
    with pytest.raises(ValueError, match="需要prompt"):
        _build_prompt("point", None, multiple=True, output="point")


def test_parse_locate_output_keeps_labels_and_no_fake_confidence():
    raw = (
        "<ref>car</ref><box><100><200><400><600></box>"
        "<box><500><300><900><800></box><ref>button</ref><box><1200><-5></box>"
    )
    boxes, points, warnings = parse_locate_output(raw, (100, 200))
    assert [box.label for box in boxes] == ["car", "car"]
    assert boxes[0].xyxy == pytest.approx((20, 20, 80, 60))
    assert points[0].xy == pytest.approx((200, 0))
    assert warnings and not hasattr(boxes[0], "confidence")


def test_locate_result_json_and_plot(tmp_path):
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    boxes, points, warnings = parse_locate_output("<ref>x</ref><box><100><100><900><900></box>", image.shape[:2])
    result = LocateAnythingResult(image, "sample.jpg", boxes=boxes, points=points, parse_warnings=warnings)
    payload = json.loads(result.to_json())
    assert payload["predictions"][0]["type"] == "box"
    assert result.plot().shape == image.shape
    assert (tmp_path / "out.jpg").as_posix() == result.save(tmp_path / "out.jpg")


def test_yolo_conversation_conversion_is_deterministic():
    classes = np.array([[1], [0]], dtype=np.float32)
    boxes = np.array([[0.5, 0.5, 0.2, 0.4], [0.25, 0.25, 0.1, 0.1]], dtype=np.float32)
    names = {0: "person", 1: "car", 2: "bus"}
    first = format_yolo_conversation("a.jpg", classes, boxes, names, seed=7)
    second = format_yolo_conversation("a.jpg", classes, boxes, names, seed=7)
    assert first == second
    assert "person</c>car</c>bus" in first["conversations"][0]["value"]
    answer = first["conversations"][1]["value"]
    assert answer.startswith("<ref>person</ref><box><200><200><300><300></box>")
    assert "<ref>car</ref><box><400><300><600><700></box>" in answer


def test_yolo_background_uses_none():
    sample = format_yolo_conversation(
        "background.jpg",
        np.empty((0, 1), dtype=np.float32),
        np.empty((0, 4), dtype=np.float32),
        {0: "person", 1: "car"},
        seed=3,
    )
    assert sample["conversations"][1]["value"] == "<box>none</box>"


def test_yolo_negative_classes_never_exceed_positive_classes():
    sample = format_yolo_conversation(
        "one.jpg",
        np.array([[0]], dtype=np.float32),
        np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32),
        {0: "person", 1: "car", 2: "bus", 3: "bike"},
        seed=4,
        negative_ratio=99,
    )
    categories = sample["conversations"][0]["value"].split("description: ", 1)[1].rstrip(".").split("</c>")
    assert len(categories) == 2


def test_conversation_jsonl_uses_utf8_byte_offsets_and_rejects_video(tmp_path):
    annotation = tmp_path / "训练.jsonl"
    records = [
        {
            "conversations": [{"from": "human", "value": "你好"}, {"from": "gpt", "value": "世界"}],
            "image": "一.jpg",
        },
        {
            "conversations": [{"from": "human", "value": "视频"}, {"from": "gpt", "value": "不支持"}],
            "video": "a.mp4",
        },
    ]
    annotation.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records), encoding="utf-8")
    dataset = ConversationDataset(annotation)
    assert dataset[0]["conversations"][0]["value"] == "你好"
    assert dataset[0]["images"] == [str(tmp_path / "一.jpg")]
    with pytest.raises(ValueError, match="不支持视频"):
        dataset[1]


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def __init__(self):
        self.ids = {
            "<|im_start|>": 1,
            "<|im_end|>": 2,
            "assistant": 3,
            "<text_mask>": 4,
            "<null>": 5,
            "</box>": 6,
            "</ref>": 7,
        }

    def convert_tokens_to_ids(self, token):
        return self.ids[token]

    def encode(self, text, add_special_tokens=False):
        return [self.ids[text]]


class _TinyLanguage(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2)
        self.config = SimpleNamespace(use_cache=True)

    def save_pretrained(self, path, **kwargs):
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter.json").write_text("{}", encoding="utf-8")


class _TinyLocateModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(use_cache=True)
        self.language_model = _TinyLanguage()
        self.vision_model = torch.nn.Linear(2, 2)
        self.mlp1 = torch.nn.Linear(2, 2)

    def gradient_checkpointing_enable(self, kwargs):
        self.gradient_checkpointing_kwargs = kwargs

    def wrap_llm_lora(self, **kwargs):
        self.lora_kwargs = kwargs
        for parameter in self.language_model.parameters():
            parameter.requires_grad = True

    def forward(self, pixel_values, **kwargs):
        hidden = self.mlp1(pixel_values.float())
        return SimpleNamespace(loss=self.language_model.proj(hidden).square().mean())


class _TinyProcessor:
    tokenizer = object()

    def save_pretrained(self, path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "processor.json").write_text("{}", encoding="utf-8")


class _TinyCollator:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, samples):
        return {
            "input_ids": torch.ones((1, 2), dtype=torch.long),
            "labels": torch.ones((1, 2), dtype=torch.long),
            "position_ids": torch.arange(2).unsqueeze(0),
            "attention_mask": torch.ones((1, 2), dtype=torch.bool),
            "pixel_values": torch.ones((1, 2)),
            "image_grid_hws": torch.ones((1, 2), dtype=torch.int32),
            "image_flags": torch.ones(1, dtype=torch.long),
        }


def _tiny_owner():
    return SimpleNamespace(
        model=_TinyLocateModel(),
        processor=_TinyProcessor(),
        model_name="fake/LocateAnything",
        revision="c32291ca5e996f5a7a485845b4f57a233936bba0",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_build_pbd_labels_appends_aligned_blocks():
    tokenizer = _FakeTokenizer()
    ids = torch.tensor([1, 3, 8, 10, 11, 7, 12, 13, 14, 15, 16, 6, 2])
    encoded = build_pbd_labels(ids, tokenizer, max_length=64)
    assert len(encoded["input_ids"]) == len(encoded["labels"]) == len(encoded["position_ids"])
    assert (encoded["input_ids"] == tokenizer.ids["<text_mask>"]).any()
    assert (encoded["labels"] != -100).any()


def test_build_pbd_labels_text_only_is_deterministic():
    tokenizer = _FakeTokenizer()
    ids = torch.tensor([1, 3, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2])
    first = build_pbd_labels(ids, tokenizer, max_length=64, seed=9)
    second = build_pbd_labels(ids, tokenizer, max_length=64, seed=9)
    assert torch.equal(first["input_ids"], second["input_ids"])
    assert (first["input_ids"] == tokenizer.ids["<text_mask>"]).any()
    assert len(first["input_ids"]) <= 64


def test_native_lora_loop_checkpoint_resume_and_export(tmp_path, monkeypatch):
    monkeypatch.setattr("ultralytics.models.locateanything.train.build_locate_dataset", lambda *args, **kwargs: [0])
    monkeypatch.setattr("ultralytics.models.locateanything.train.LocateAnythingCollator", _TinyCollator)
    output = tmp_path / "run"
    events = []
    callback_map = defaultdict(list)
    expected_events = (
        "on_pretrain_routine_start",
        "on_pretrain_routine_end",
        "on_train_start",
        "on_train_epoch_start",
        "on_train_batch_start",
        "optimizer_step",
        "on_before_zero_grad",
        "on_params_update",
        "on_train_batch_end",
        "on_train_epoch_end",
        "on_fit_epoch_end",
        "on_model_save",
        "on_train_end",
        "teardown",
    )
    for event in expected_events:
        callback_map[event].append(lambda trainer, event=event: events.append(event))
    first = LocateAnythingTrainer(
        model=_tiny_owner(),
        data="fake.jsonl",
        output_dir=output,
        epochs=1,
        max_steps=1,
        workers=0,
        max_seq_length=128,
        save_steps=1,
        callbacks_=callback_map,
    ).train()
    assert first.steps == 1
    assert (output / "final" / "llm_adapter" / "adapter.json").is_file()
    assert json.loads((output / "final" / "locateanything.json").read_text())["artifact"] == "lora"
    checkpoint = Path(first.last_checkpoint)
    assert json.loads((checkpoint / "trainer.json").read_text())["epoch"] == 1
    assert set(expected_events).issubset(events)

    second = LocateAnythingTrainer(
        model=_tiny_owner(),
        data="fake.jsonl",
        output_dir=output,
        epochs=2,
        max_steps=2,
        workers=0,
        max_seq_length=128,
        save_steps=1,
        resume=True,
    ).train()
    assert second.steps == 2
    assert Path(second.last_checkpoint).name == "step-2"


def test_shared_callback_host():
    host = type("Host", (CallbackHost,), {})()
    host.callbacks = defaultdict(list)
    calls = []
    host.add_callback("on_train_start", lambda owner: calls.append(owner))
    host.run_callbacks("on_train_start")
    assert calls == [host]


def test_lora_embedding_output_is_nonleaf_for_visual_token_replacement():
    embedding = torch.nn.Embedding(4, 2)
    embedding.weight.requires_grad_(False)
    embedding.register_forward_hook(lambda _module, _inputs, output: output.requires_grad_(True))
    embedding.register_forward_hook(_clone_embedding_output)
    output = embedding(torch.tensor([0, 1]))
    assert output.requires_grad and not output.is_leaf
    output[0] = torch.ones(2)
    output.sum().backward()


def test_lora_disables_only_remote_auxiliary_training_return():
    child = torch.nn.Linear(2, 2)
    language_model = torch.nn.Sequential(child)
    language_model.train()
    _disable_remote_auxiliary_training_return(language_model)
    assert not language_model.training
    assert child.training


def test_transformers_514_compat_metadata_patch():
    transformers = pytest.importorskip("transformers")

    class FakeConfig(transformers.PretrainedConfig):
        pass

    class FakeLegacyModel(transformers.PreTrainedModel):
        config_class = FakeConfig
        _tied_weights_keys = ["lm_head.weight"]

        def _check_and_adjust_attn_implementation(self, attn_implementation, is_init_check=False):
            return attn_implementation

    config = SimpleNamespace(
        text_config=SimpleNamespace(rope_parameters={"rope_theta": 1_000_000.0}),
    )
    patch_transformers_514(FakeLegacyModel, config)
    assert FakeLegacyModel._tied_weights_keys == {"lm_head.weight": "model.embed_tokens.weight"}
    assert config.text_config.rope_theta == 1_000_000.0
    assert "kwargs" in inspect.signature(FakeLegacyModel._check_and_adjust_attn_implementation).parameters
    legacy = ((torch.ones(1, 1, 2, 1), torch.zeros(1, 1, 2, 1)),)
    cache = transformers.DynamicCache.from_legacy_cache(legacy)
    restored = cache.to_legacy_cache()
    assert len(restored) == 1 and torch.equal(restored[0][0], legacy[0][0])


def test_transformers_514_generation_and_return_dict_patches_are_idempotent(caplog):
    transformers = pytest.importorskip("transformers")
    from transformers.generation import GenerationMixin

    class FakeConfig(transformers.PretrainedConfig):
        pass

    Qwen2ForCausalLM = type(
        "Qwen2ForCausalLM",
        (transformers.PreTrainedModel,),
        {"config_class": FakeConfig, "prepare_inputs_for_generation": lambda self, input_ids: {"input_ids": input_ids}},
    )
    config = FakeConfig(return_dict=False)
    config.text_config = FakeConfig(return_dict=False)
    config.vision_config = FakeConfig(return_dict=True)
    patch_transformers_514(Qwen2ForCausalLM, config)
    patch_transformers_514(Qwen2ForCausalLM, config)

    caplog.clear()
    assert issubclass(Qwen2ForCausalLM, GenerationMixin)
    assert Qwen2ForCausalLM.__bases__.count(GenerationMixin) == 1
    assert config.text_config.use_return_dict is False
    assert config.vision_config.use_return_dict is True
    assert "use_return_dict" not in caplog.text


def test_transformers_514_processor_uses_auto_mapping():
    class FakeProcessor:
        image_processor_class = "AutoImageProcessor"

    _patch_processor_class(FakeProcessor)
    _patch_processor_class(FakeProcessor)
    assert "image_processor_class" not in FakeProcessor.__dict__


def test_transformers_514_attention_mask_matches_legacy_without_warning():
    from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask as legacy_mask

    inputs_embeds = torch.zeros((1, 3, 8), dtype=torch.float32)
    attention_mask = torch.tensor([[0, 1, 1, 1, 1]], dtype=torch.long)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        expected = legacy_mask(attention_mask, (1, 3), inputs_embeds, 2, sliding_window=4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = _prepare_4d_causal_attention_mask_compat(attention_mask, (1, 3), inputs_embeds, 2, sliding_window=4)
    assert torch.equal(actual, expected)
    assert caught == []

    allowed_4d = torch.tril(torch.ones((1, 1, 3, 3), dtype=torch.float32))
    actual_4d = _prepare_4d_causal_attention_mask_compat(allowed_4d, (1, 3), inputs_embeds, 0)
    expected_4d = (1.0 - allowed_4d).masked_fill((1.0 - allowed_4d).bool(), torch.finfo(torch.float32).min)
    assert torch.equal(actual_4d, expected_4d)


def test_transformers_514_tensor_copy_proxy_clones_without_warning():
    proxy = _TorchTensorCopyCompat(torch)
    source = torch.tensor([1, 2], dtype=torch.int64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        copied = proxy.tensor(source, dtype=torch.float32, device="cpu")
    assert caught == []
    assert copied.dtype == torch.float32
    assert copied.data_ptr() != source.data_ptr()
    assert torch.equal(copied, source.float())


def test_transformers_514_moonvit_rope_precompute_is_numerically_stable():
    rope = SimpleNamespace(max_height=3, max_width=2, dim=8, theta_base=10_000.0)
    actual = _moonvit_rope_precompute_compat(rope, torch.device("cpu"))
    flat_pos = torch.arange(6, dtype=torch.float32)
    dim_range = torch.arange(0, 8, 4, dtype=torch.float32)
    freqs = 1.0 / (rope.theta_base ** (dim_range / rope.dim))
    x_freqs = torch.outer(flat_pos % rope.max_width, freqs)
    y_freqs = torch.outer(flat_pos // rope.max_width, freqs)
    expected = torch.cat(
        [
            torch.polar(torch.ones_like(x_freqs), x_freqs).unsqueeze(-1),
            torch.polar(torch.ones_like(y_freqs), y_freqs).unsqueeze(-1),
        ],
        dim=-1,
    ).reshape(rope.max_height, rope.max_width, -1)
    torch.testing.assert_close(actual, expected)


def test_transformers_514_repairs_meta_materialized_rope_buffers():
    class Qwen2RotaryEmbedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dim = 8
            self.base = 10_000.0
            self.max_position_embeddings = 16
            self.register_buffer("inv_freq", torch.zeros(4), persistent=False)
            self.register_buffer("cos_cached", torch.zeros(16, 8), persistent=False)
            self.register_buffer("sin_cached", torch.zeros(16, 8), persistent=False)

    model = torch.nn.Sequential(Qwen2RotaryEmbedding(), Qwen2RotaryEmbedding())
    assert _repair_qwen_rope_buffers(model) == 2
    for rotary in model:
        assert torch.equal(rotary.cos_cached[0], torch.ones(8))
        assert torch.equal(rotary.sin_cached[0], torch.zeros(8))
        assert rotary.inv_freq.count_nonzero() == 4


@pytest.mark.parametrize("device_type", ["cuda", "npu"])
def test_fsdp2_backend_branch_and_bottom_up_wrapping(monkeypatch, device_type):
    language_layers = [torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)]
    vision_blocks = [torch.nn.Linear(2, 2)]
    root = SimpleNamespace(
        language_model=SimpleNamespace(model=SimpleNamespace(layers=language_layers)),
        vision_model=SimpleNamespace(encoder=SimpleNamespace(blocks=vision_blocks)),
    )
    trainer = object.__new__(LocateAnythingTrainer)
    trainer.device = torch.device(device_type, 0)
    trainer.world_size = 2
    trainer.model = root
    calls = []

    def fake_fully_shard(module, **kwargs):
        calls.append((module, kwargs))

    monkeypatch.setattr("torch.distributed.device_mesh.init_device_mesh", lambda *args, **kwargs: "mesh")
    if device_type == "npu":
        monkeypatch.setattr("torch_npu.distributed.fsdp.fully_shard", fake_fully_shard)
    else:
        monkeypatch.setattr("torch.distributed.fsdp.fully_shard", fake_fully_shard)
    trainer._apply_fsdp2()
    assert [module for module, _ in calls] == [*language_layers, *vision_blocks, root]
    assert all(kwargs["mesh"] == "mesh" for _, kwargs in calls)
