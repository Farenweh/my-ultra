#!/usr/bin/env python3
"""运行LocateAnything 8卡batch验证基准并输出结构化吞吐结果。"""

from __future__ import annotations

import argparse
import json

from ultralytics import LocateAnything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="nvidia/LocateAnything-3B")
    parser.add_argument("--data", default="coco.yaml")
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-images", type=int, default=128)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--continuous-window", type=int, default=1)
    parser.add_argument("--continuous-batching", action="store_true")
    parser.add_argument("--dynamic-scheduling", action="store_true")
    parser.add_argument("--refill-batch", type=int, default=0)
    parser.add_argument("--static-kv-cache", action="store_true")
    parser.add_argument("--paged-kv-cache", action="store_true")
    parser.add_argument("--shape-bucketing", action="store_true")
    parser.add_argument("--kv-bucket-size", type=int, default=128)
    parser.add_argument("--npu-graph", action="store_true")
    parser.add_argument("--visual-batching", action="store_true")
    parser.add_argument("--max-duplicate-boxes", type=int, default=0)
    parser.add_argument("--no-cpu-affinity", action="store_true")
    parser.add_argument("--npu-fast-path", choices=("auto", "off", "strict"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = LocateAnything(args.model, local_files_only=True, npu_fast_path=args.npu_fast_path)
    metrics = model.val(
        data=args.data,
        device=args.devices,
        batch=args.batch,
        max_new_tokens=args.max_new_tokens,
        continuous_window=args.continuous_window,
        continuous_batching=args.continuous_batching,
        dynamic_scheduling=args.dynamic_scheduling,
        refill_batch=args.refill_batch,
        static_kv_cache=args.static_kv_cache,
        paged_kv_cache=args.paged_kv_cache,
        shape_bucketing=args.shape_bucketing,
        kv_bucket_size=args.kv_bucket_size,
        npu_graph=args.npu_graph,
        visual_batching=args.visual_batching,
        max_duplicate_boxes=args.max_duplicate_boxes,
        cpu_affinity=not args.no_cpu_affinity,
        max_images=args.max_images,
        output_dir=args.output,
    )
    print(json.dumps({"results": metrics.results_dict, "speed": metrics.speed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
