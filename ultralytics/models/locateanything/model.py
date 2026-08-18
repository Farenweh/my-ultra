# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Ultralytics原生LocateAnything Python接口。"""

from __future__ import annotations

import time
from queue import Empty, Full, Queue
from threading import Event, Thread
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
import torch
from PIL import Image

from ultralytics.engine.runtime import CallbackHost
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import parse_device, select_device

from .compat import DEFAULT_MODEL, SUPPORTED_REVISION, load_locate_components, resolve_dtype
from .results import LocateAnythingResult, parse_locate_output


class _ContinuousBatchPrefetcher:
    """在单个后台线程中预处理后续图片，NPU搬运仍由生成主线程执行。"""

    _END = object()

    def __init__(
        self,
        source_provider: Callable[[int], list[tuple[Any, int, Any] | tuple[Any, int, Any, str]]],
        prepare: Callable[[Any, str], tuple[Any, tuple[np.ndarray, str], float]],
        question: str | None,
        *,
        total: int,
        request_size: int,
        capacity: int,
    ) -> None:
        self.source_provider = source_provider
        self.prepare = prepare
        self.question = question
        self.total = total
        self.request_size = max(1, request_size)
        self.queue: Queue = Queue(maxsize=max(1, capacity))
        self.stop_event = Event()
        self.exhausted = False
        self.thread = Thread(target=self._run, name="locate-preprocess", daemon=True)
        self.thread.start()

    def _put(self, item: Any) -> bool:
        while not self.stop_event.is_set():
            try:
                self.queue.put(item, timeout=0.1)
                return True
            except Full:
                continue
        return False

    def _run(self) -> None:
        produced = 0
        try:
            while produced < self.total and not self.stop_event.is_set():
                request = min(self.request_size, self.total - produced)
                items = self.source_provider(request)
                for item in items:
                    if len(item) == 4:
                        source, seed, context, question = item
                    else:
                        source, seed, context = item
                        question = self.question
                    if question is None:
                        raise ValueError("LocateAnything批量预处理缺少逐样本prompt")
                    prepared, original, elapsed_ms = self.prepare(source, question)
                    if not self._put((prepared, original, elapsed_ms, int(seed), context)):
                        return
                    produced += 1
                if len(items) < request:
                    break
        except BaseException as error:
            self._put(error)
        finally:
            self._put(self._END)

    def get(self, count: int) -> list[tuple[Any, tuple[np.ndarray, str], float, int, Any]]:
        """按调度器请求返回至多count个已完成CPU预处理的样本。"""
        items = []
        while len(items) < count and not self.exhausted:
            item = self.queue.get()
            if item is self._END:
                self.exhausted = True
                break
            if isinstance(item, BaseException):
                self.exhausted = True
                raise RuntimeError("LocateAnything后台图片预处理失败") from item
            items.append(item)
        return items

    def close(self) -> None:
        self.stop_event.set()
        while self.thread.is_alive():
            try:
                self.queue.get_nowait()
            except Empty:
                break
        self.thread.join(timeout=5)


class LocateAnything(CallbackHost):
    """NVIDIA LocateAnything的推理与微调入口。"""

    def __init__(
        self,
        model: str | Path = DEFAULT_MODEL,
        *,
        revision: str = SUPPORTED_REVISION,
        device: str | int | torch.device | None = None,
        dtype: str | torch.dtype | None = "auto",
        local_files_only: bool = False,
        npu_fast_path: str | bool | None = "auto",
    ) -> None:
        self.model_name = str(model)
        self.revision = revision
        self.device = _resolve_device(device)
        self.dtype = resolve_dtype(dtype, self.device)
        self.local_files_only = local_files_only
        from .npu_fast import normalize_npu_fast_policy

        self.npu_fast_path = normalize_npu_fast_policy(npu_fast_path)
        self.metrics = None
        self.setup_callbacks()
        self.model, self.processor, self.tokenizer = load_locate_components(
            self.model_name,
            revision=revision,
            device=self.device,
            dtype=self.dtype,
            attn_implementation="sdpa",
            local_files_only=local_files_only,
            npu_fast_path=self.npu_fast_path,
        )

    def __call__(self, source: Any, **kwargs: Any):
        """调用 :meth:`predict`。"""
        return self.predict(source, **kwargs)

    def predict(
        self,
        source: Any,
        *,
        task: str = "ground",
        prompt: str | list[str] | tuple[str, ...] | None = None,
        multiple: bool = True,
        output: str = "box",
        generation_mode: str | None = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stream: bool = False,
        verbose: bool = False,
    ) -> list[LocateAnythingResult] | Iterator[LocateAnythingResult]:
        """在一张或多张图片上执行定位。"""
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens必须大于0")
        question, default_label = _build_prompt(task, prompt, multiple=multiple, output=output)
        mode = generation_mode or ("slow" if self.device.type == "npu" else "hybrid")
        if mode not in {"fast", "slow", "hybrid"}:
            raise ValueError("generation_mode必须是'fast'、'slow'或'hybrid'")
        sources = source if isinstance(source, (list, tuple)) else [source]
        generator = (
            self._predict_one(
                item,
                question,
                default_label=default_label,
                generation_mode=mode,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                verbose=verbose,
            )
            for item in sources
        )
        return generator if stream else list(generator)

    def detect(self, source: Any, classes: str | list[str] | tuple[str, ...], **kwargs: Any):
        """定位给定类别的所有实例。"""
        return self.predict(source, task="detect", prompt=classes, **kwargs)

    def ground(self, source: Any, phrase: str, *, multiple: bool = True, **kwargs: Any):
        """按自由短语定位一个或多个实例。"""
        return self.predict(source, task="ground", prompt=phrase, multiple=multiple, **kwargs)

    def ground_text(self, source: Any, text: str, **kwargs: Any):
        """定位指定文字。"""
        return self.predict(source, task="ground_text", prompt=text, **kwargs)

    def detect_text(self, source: Any, **kwargs: Any):
        """定位图像中的全部文字区域。"""
        return self.predict(source, task="detect_text", **kwargs)

    def ground_gui(self, source: Any, phrase: str, *, output: str = "box", **kwargs: Any):
        """将GUI描述定位为框或点。"""
        return self.predict(source, task="gui", prompt=phrase, output=output, **kwargs)

    def point(self, source: Any, phrase: str, **kwargs: Any):
        """返回目标内部的定位点。"""
        return self.predict(source, task="point", prompt=phrase, output="point", **kwargs)

    def train(self, data: str | Path, *, method: str = "lora", **kwargs: Any):
        """使用LocateAnything专用训练器执行LoRA或全参数SFT。"""
        from .npu_fast import release_npu_inference_caches
        from .npu_graph import configure_npu_graph
        from .train import LocateAnythingTrainer

        configure_npu_graph(self.model, False)
        release_npu_inference_caches(self.model)
        trainer = LocateAnythingTrainer(
            model=self,
            data=data,
            method=method,
            callbacks_=self.callbacks,
            **kwargs,
        )
        return trainer.train()

    def val(
        self,
        data: str | Path = "coco.yaml",
        *,
        device: str | None = None,
        batch: int = 1,
        scheduler: str = "pipeline",
        protocol: str = "paper",
        **kwargs: Any,
    ):
        """使用LocateAnything专用validator执行MS COCO验证。"""
        from .val import LocateAnythingValidator

        validator = LocateAnythingValidator(
            model=self,
            data=data,
            device=device,
            batch=batch,
            scheduler=scheduler,
            protocol=protocol,
            callbacks_=self.callbacks,
            **kwargs,
        )
        self.metrics = validator()
        return self.metrics

    @torch.no_grad()
    def _predict_one(
        self,
        source: Any,
        question: str,
        *,
        default_label: str,
        generation_mode: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        verbose: bool,
    ) -> LocateAnythingResult:
        """处理单张图片并返回结构化结果。"""
        preprocess_start = time.perf_counter()
        pil_image, orig_img, path = _load_image(source)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = self.processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos, return_tensors="pt").to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        _synchronize(self.device)
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000

        inference_start = time.perf_counter()
        response = self.model.generate(
            pixel_values=inputs.get("pixel_values"),
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            image_grid_hws=inputs.get("image_grid_hws"),
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode=generation_mode,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            verbose=verbose,
        )
        _synchronize(self.device)
        inference_ms = (time.perf_counter() - inference_start) * 1000
        answer, upstream_stats = _unpack_response(response)
        stats = {"output_tokens": _count_output_tokens(self.tokenizer, answer)}
        if upstream_stats is not None:
            stats["upstream"] = upstream_stats

        postprocess_start = time.perf_counter()
        boxes, points, warnings = parse_locate_output(answer, orig_img.shape[:2], default_label=default_label)
        postprocess_ms = (time.perf_counter() - postprocess_start) * 1000
        return LocateAnythingResult(
            orig_img,
            path,
            boxes=boxes,
            points=points,
            raw_output=answer,
            parse_warnings=warnings,
            stats=stats,
            speed={"preprocess": preprocess_ms, "inference": inference_ms, "postprocess": postprocess_ms},
        )

    @torch.no_grad()
    def _prepare_batch_source_cpu(self, source: Any, question: str):
        """在CPU把一张图片转换为hybrid输入，不在后台线程中访问NPU。"""
        from .batch import BatchInput

        start = time.perf_counter()
        pil_image, orig_img, path = _load_image(source)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = self.processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos, return_tensors="pt")
        prepared = BatchInput(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            image_grid_hws=inputs.get("image_grid_hws"),
        )
        return prepared, (orig_img, path), (time.perf_counter() - start) * 1000

    def _move_batch_input_to_device(self, prepared: Any):
        """由生成主线程把预处理结果搬到目标设备。"""
        from .batch import BatchInput

        grid = prepared.image_grid_hws
        if grid is not None:
            grid = torch.as_tensor(grid, device=self.device)
        return BatchInput(
            input_ids=prepared.input_ids.to(self.device),
            pixel_values=prepared.pixel_values.to(device=self.device, dtype=self.dtype),
            image_grid_hws=grid,
        )

    def _prepare_batch_source(self, source: Any, question: str):
        """同步预处理入口，供普通批量预测保持原行为。"""
        start = time.perf_counter()
        prepared, original, _ = self._prepare_batch_source_cpu(source, question)
        prepared = self._move_batch_input_to_device(prepared)
        return prepared, original, (time.perf_counter() - start) * 1000

    @torch.no_grad()
    def _predict_batch(
        self,
        sources: list[Any],
        question: str | list[str],
        *,
        default_label: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        scheduler: str,
        seeds: list[int],
        batch_id: int,
        slot_capacity: int | None = None,
        static_kv_cache: bool = False,
        paged_kv_cache: bool = False,
        max_duplicate_boxes: int = 0,
        shape_bucketing: bool = False,
        kv_bucket_size: int = 128,
        npu_graph: bool = False,
        visual_batching: bool = False,
        direct_paged_decode: bool = True,
        device_repetition_cache: bool = True,
        qsample_reservoir: bool = False,
        overlap_prefill: bool = True,
        candidate_top_p: bool = True,
    ) -> list[LocateAnythingResult]:
        """使用SDPA hybrid runtime执行一个真实批量，仅供显式batch>1的validator调用。"""
        from .batch import BatchInput, generate_batch_hybrid

        if len(sources) != len(seeds):
            raise ValueError("sources与seeds数量必须一致")
        questions = [question] * len(sources) if isinstance(question, str) else list(question)
        if len(questions) != len(sources):
            raise ValueError("sources与questions数量必须一致")
        prepared: list[BatchInput] = []
        originals: list[tuple[np.ndarray, str]] = []
        preprocess_ms: list[float] = []
        for source, sample_question in zip(sources, questions):
            item, original, elapsed_ms = self._prepare_batch_source(source, sample_question)
            prepared.append(item)
            originals.append(original)
            preprocess_ms.append(elapsed_ms)

        _synchronize(self.device)
        generation_start = time.perf_counter()
        outputs = generate_batch_hybrid(
            self.model,
            self.tokenizer,
            prepared,
            device=self.device,
            dtype=self.dtype,
            seeds=seeds,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            scheduler=scheduler,
            slot_capacity=slot_capacity,
            static_kv_cache=static_kv_cache,
            paged_kv_cache=paged_kv_cache,
            max_duplicate_boxes=max_duplicate_boxes,
            shape_bucketing=shape_bucketing,
            kv_bucket_size=kv_bucket_size,
            npu_graph=npu_graph,
            visual_batching=visual_batching,
            direct_paged_decode=direct_paged_decode,
            device_repetition_cache=device_repetition_cache,
            qsample_reservoir=qsample_reservoir,
            overlap_prefill=overlap_prefill,
            candidate_top_p=candidate_top_p,
        )
        _synchronize(self.device)
        generation_seconds = time.perf_counter() - generation_start
        batch_output_tokens = sum(output.output_tokens for output in outputs)
        effective_capacity = min(slot_capacity or len(sources), len(sources))
        results = []
        for row, (output, (orig_img, path)) in enumerate(zip(outputs, originals)):
            postprocess_start = time.perf_counter()
            boxes, points, warnings = parse_locate_output(output.text, orig_img.shape[:2], default_label=default_label)
            postprocess_ms = (time.perf_counter() - postprocess_start) * 1000
            results.append(
                LocateAnythingResult(
                    orig_img,
                    path,
                    boxes=boxes,
                    points=points,
                    raw_output=output.text,
                    parse_warnings=warnings,
                    stats={
                        "output_tokens": output.output_tokens,
                        "batch_id": int(batch_id),
                        "batch_size": effective_capacity,
                        "continuous_window_size": len(sources),
                        "batch_output_tokens": batch_output_tokens,
                        "batch_generation_seconds": generation_seconds,
                        "forward_steps": output.forward_steps,
                        "switch_to_ar": output.switch_to_ar,
                        "stopped_repetition": output.stopped_repetition,
                        "seed": int(seeds[row]),
                    },
                    speed={
                        "preprocess": preprocess_ms[row],
                        "inference": generation_seconds * 1000,
                        "postprocess": postprocess_ms,
                    },
                )
            )
        return results

    @torch.no_grad()
    def _predict_continuous(
        self,
        source_provider: Callable[[int], list[tuple[Any, int, Any] | tuple[Any, int, Any, str]]],
        result_callback: Callable[[Any, LocateAnythingResult], None],
        question: str | None,
        *,
        default_label: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        scheduler: str,
        slot_capacity: int,
        refill_batch_size: int,
        max_provider_inputs: int,
        preprocess_request_size: int | None = None,
        paged_kv_cache: bool = False,
        max_duplicate_boxes: int = 0,
        shape_bucketing: bool = False,
        kv_bucket_size: int = 128,
        npu_graph: bool = False,
        visual_batching: bool = False,
        direct_paged_decode: bool = True,
        device_repetition_cache: bool = True,
        qsample_reservoir: bool = False,
        overlap_prefill: bool = True,
        candidate_top_p: bool = True,
    ) -> dict[str, float | int]:
        """持续补充活跃槽位，并在样本完成时立即回调写出结果。"""
        from .batch import BatchInput, BatchOutput, generate_batch_hybrid

        originals: list[tuple[np.ndarray, str] | None] = []
        contexts: list[Any | None] = []
        sample_seeds: list[int] = []
        preprocess_ms: list[float] = []
        admitted_at: list[float] = []
        generation_start = time.perf_counter()
        completed_tokens = 0
        provider_seconds = 0.0
        callback_seconds = 0.0
        prefetcher = _ContinuousBatchPrefetcher(
            source_provider,
            self._prepare_batch_source_cpu,
            question,
            total=max_provider_inputs,
            request_size=preprocess_request_size or max(refill_batch_size, min(slot_capacity, 32)),
            capacity=max(2 * refill_batch_size, 16),
        )

        def provide(count: int) -> tuple[list[BatchInput], list[int]]:
            nonlocal provider_seconds
            provider_start = time.perf_counter()
            items = prefetcher.get(count)
            prepared, seeds = [], []
            for batch_input, original, elapsed_ms, seed, context in items:
                transfer_start = time.perf_counter()
                batch_input = self._move_batch_input_to_device(batch_input)
                elapsed_ms += (time.perf_counter() - transfer_start) * 1000
                prepared.append(batch_input)
                seeds.append(int(seed))
                originals.append(original)
                contexts.append(context)
                sample_seeds.append(int(seed))
                preprocess_ms.append(elapsed_ms)
                admitted_at.append(time.perf_counter())
            provider_seconds += time.perf_counter() - provider_start
            return prepared, seeds

        def complete(rows: list[tuple[int, BatchOutput]]) -> None:
            nonlocal callback_seconds, completed_tokens
            callback_start = time.perf_counter()
            scheduler_elapsed = max(time.perf_counter() - generation_start - provider_seconds - callback_seconds, 0.0)
            for row, output in rows:
                original = originals[row]
                context = contexts[row]
                if original is None or context is None:
                    raise RuntimeError(f"continuous batching第{row}行缺少结果上下文")
                orig_img, path = original
                postprocess_start = time.perf_counter()
                boxes, points, warnings = parse_locate_output(
                    output.text, orig_img.shape[:2], default_label=default_label
                )
                postprocess_ms = (time.perf_counter() - postprocess_start) * 1000
                completed_tokens += output.output_tokens
                batch_id = int(context.get("id", row)) if isinstance(context, dict) else row
                result = LocateAnythingResult(
                    orig_img,
                    path,
                    boxes=boxes,
                    points=points,
                    raw_output=output.text,
                    parse_warnings=warnings,
                    stats={
                        "output_tokens": output.output_tokens,
                        "batch_id": batch_id,
                        "batch_size": min(slot_capacity, max_provider_inputs),
                        "continuous_window_size": 0,
                        "batch_output_tokens": output.output_tokens,
                        "batch_generation_seconds": max(time.perf_counter() - admitted_at[row], 0.0),
                        "scheduler_output_tokens": completed_tokens,
                        "scheduler_generation_seconds": scheduler_elapsed,
                        "continuous_refill_batch": refill_batch_size,
                        "forward_steps": output.forward_steps,
                        "switch_to_ar": output.switch_to_ar,
                        "stopped_repetition": output.stopped_repetition,
                        "seed": sample_seeds[row],
                    },
                    speed={
                        "preprocess": preprocess_ms[row],
                        "inference": max(time.perf_counter() - admitted_at[row], 0.0) * 1000,
                        "postprocess": postprocess_ms,
                    },
                )
                result_callback(context, result)
                originals[row] = None
                contexts[row] = None
            callback_seconds += time.perf_counter() - callback_start

        _synchronize(self.device)
        generation_start = time.perf_counter()
        try:
            outputs = generate_batch_hybrid(
                self.model,
                self.tokenizer,
                [],
                device=self.device,
                dtype=self.dtype,
                seeds=[],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                scheduler=scheduler,
                slot_capacity=slot_capacity,
                paged_kv_cache=paged_kv_cache,
                max_duplicate_boxes=max_duplicate_boxes,
                input_provider=provide,
                completion_callback=complete,
                refill_batch_size=refill_batch_size,
                max_provider_inputs=max_provider_inputs,
                shape_bucketing=shape_bucketing,
                kv_bucket_size=kv_bucket_size,
                npu_graph=npu_graph,
                visual_batching=visual_batching,
                direct_paged_decode=direct_paged_decode,
                device_repetition_cache=device_repetition_cache,
                qsample_reservoir=qsample_reservoir,
                overlap_prefill=overlap_prefill,
                candidate_top_p=candidate_top_p,
            )
        finally:
            prefetcher.close()
        _synchronize(self.device)
        total_seconds = time.perf_counter() - generation_start
        generation_seconds = max(total_seconds - provider_seconds - callback_seconds, 0.0)
        return {
            "processed": len(outputs),
            "output_tokens": sum(output.output_tokens for output in outputs),
            "generation_seconds": generation_seconds,
            "scheduler_overhead_seconds": provider_seconds + callback_seconds,
        }


def _build_prompt(
    task: str,
    prompt: str | list[str] | tuple[str, ...] | None,
    *,
    multiple: bool,
    output: str,
) -> tuple[str, str]:
    """构造官方任务模板和无ref输出时的默认标签。"""
    task = task.lower().strip()
    task = {"ground_gui": "gui", "detection": "detect"}.get(task, task)
    if output not in {"box", "point"}:
        raise ValueError("output必须是'box'或'point'")
    if task == "detect_text":
        return "Detect all the text in box format.", "text"
    if prompt is None:
        raise ValueError(f"task={task!r}需要prompt")
    if task == "detect":
        categories = [prompt] if isinstance(prompt, str) else list(prompt)
        categories = [str(x).strip() for x in categories if str(x).strip()]
        if not categories:
            raise ValueError("detect至少需要一个非空类别")
        joined = "</c>".join(categories)
        return f"Locate all the instances that matches the following description: {joined}.", joined
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"task={task!r}需要非空字符串prompt")
    phrase = prompt.strip()
    if task == "ground":
        quantifier = "all the instances" if multiple else "a single instance"
        return (
            f"Locate {quantifier} that match{'es' if not multiple else ''} the following description: {phrase}.",
            phrase,
        )
    if task == "ground_text":
        return f"Please locate the text referred as {phrase}.", phrase
    if task == "gui":
        return (
            f"Point to: {phrase}."
            if output == "point"
            else f"Locate the region that matches the following description: {phrase}."
        ), phrase
    if task == "point":
        return f"Point to: {phrase}.", phrase
    if task == "raw":
        return phrase, phrase
    raise ValueError("不支持task={!r}，可选detect、ground、ground_text、detect_text、gui、point、raw".format(task))


def _load_image(source: Any) -> tuple[Image.Image, np.ndarray, str]:
    """将路径、URL、PIL或NumPy输入规范成RGB PIL和BGR NumPy图像。"""
    path = "image.jpg"
    if isinstance(source, Image.Image):
        pil_image = source.convert("RGB")
        path = getattr(source, "filename", path) or path
    elif isinstance(source, np.ndarray):
        array = source
        if array.ndim != 3 or array.shape[2] not in {3, 4}:
            raise ValueError(f"NumPy图片必须是HWC三或四通道，得到{array.shape}")
        if array.shape[2] == 4:
            array = cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
        pil_image = Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))
    elif isinstance(source, (str, Path)):
        path = str(source)
        if path.startswith(("http://", "https://")):
            response = requests.get(path, timeout=30)
            response.raise_for_status()
            from io import BytesIO

            pil_image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            pil_image = Image.open(path).convert("RGB")
    else:
        raise TypeError(f"不支持图片输入类型{type(source).__name__}")
    orig_img = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
    return pil_image, orig_img, path


def _resolve_device(device: str | int | torch.device | None) -> torch.device:
    """使用Ultralytics现有设备选择逻辑解析推理设备。"""
    if isinstance(device, torch.device):
        return device
    value = parse_device(device)
    selected = select_device(value)
    if selected.type == "mps":
        LOGGER.warning("LocateAnything未验证MPS，回退到CPU")
        return torch.device("cpu")
    return selected


def _unpack_response(response: Any) -> tuple[str, Any]:
    """规范上游generate的字符串或统计元组。"""
    if isinstance(response, tuple):
        answer = response[0]
        stats = response[2] if len(response) >= 3 else None
    else:
        answer, stats = response, None
    if isinstance(answer, (list, tuple)):
        answer = answer[0]
    return str(answer), stats


def _count_output_tokens(tokenizer: Any, answer: str) -> int:
    """统计生成文本对应的模型token数，不添加额外特殊token。"""
    encoded = tokenizer(answer, add_special_tokens=False)
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if input_ids and isinstance(input_ids[0], (list, tuple)):
        input_ids = input_ids[0]
    return len(input_ids)


def _synchronize(device: torch.device) -> None:
    """在统计耗时边界同步当前加速器。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()


__all__ = ("LocateAnything",)
