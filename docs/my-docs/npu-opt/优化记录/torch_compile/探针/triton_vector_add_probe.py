from __future__ import annotations

import json
import os
import platform
import traceback

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def add_kernel(x, y, out, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    tl.store(out + offsets, tl.load(x + offsets, mask=mask) + tl.load(y + offsets, mask=mask), mask=mask)


result = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_npu": torch_npu.__version__,
    "triton_module_version": triton.__version__,
    "triton_module_path": triton.__file__,
    "visible": os.getenv("ASCEND_RT_VISIBLE_DEVICES"),
    "ascend_home": os.getenv("ASCEND_HOME_PATH"),
}
try:
    torch.npu.set_device(0)
    n = 65536
    x = torch.randn(n, device="npu:0", dtype=torch.float32)
    y = torch.randn(n, device="npu:0", dtype=torch.float32)
    out = torch.empty_like(x)
    add_kernel[(triton.cdiv(n, 256),)](x, y, out, n_elements=n, BLOCK=256)
    torch.npu.synchronize()
    x_cpu, y_cpu, out_cpu = x.cpu(), y.cpu(), out.cpu()
    expected = x_cpu + y_cpu
    result.update(
        status="success",
        max_abs_error=float((out_cpu - expected).abs().max()),
        allclose=bool(torch.allclose(out_cpu, expected)),
        output_sum=float(out_cpu.sum()),
    )
except Exception as error:
    result.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
print(json.dumps(result, ensure_ascii=False, indent=2))
