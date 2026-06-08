# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""启动带有 Ultralytics 默认界面设置的 TensorBoard。"""

from __future__ import annotations

import contextlib
import functools
import json
import re
import sys
import tempfile
import zipfile
from pathlib import Path

from tensorboard import assets, default, program

_RELOAD_PERIOD_MS = 5000
_SCALAR_SMOOTHING = 0
_BRIGHT_BLUE_COLORS = (
    ("Bright Blue", "#009dff", "#40c4ff"),
    ("Azure", "#006dff", "#4d8dff"),
    ("Cyan Blue", "#00c8ff", "#56dcff"),
    ("Sky Blue", "#2f80ff", "#7fb3ff"),
    ("Electric Blue", "#00a3ff", "#69d7ff"),
    ("Royal Blue", "#165dff", "#6f95ff"),
    ("Aqua Blue", "#00d4ff", "#72e7ff"),
)
_PALETTE_PATTERN = re.compile(
    r'colorPalette:\{id:"default",name:"Defalt",colors:\['
    r'(?:\{name:"[^"]+",lightHex:"#[0-9a-fA-F]{6}",darkHex:"#[0-9a-fA-F]{6}"\},?)+'
    r'\],inactive:\{name:"Gray",lightHex:"#[0-9a-fA-F]{6}",darkHex:"#[0-9a-fA-F]{6}"\}\}'
)


def _replace_once(text: str, old: str, new: str) -> str:
    """只替换一次，避免 TensorBoard 前端结构变化时静默失效。"""
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"TensorBoard 前端资源匹配失败：{old!r} 出现 {count} 次")
    return text.replace(old, new, 1)


def _bright_blue_palette_js() -> str:
    """生成 TensorBoard 可直接使用的亮蓝调色板对象。"""
    colors = ",".join(
        f'{{name:{json.dumps(name)},lightHex:"{light}",darkHex:"{dark}"}}'
        for name, light, dark in _BRIGHT_BLUE_COLORS
    )
    return (
        'colorPalette:{id:"ultralytics-bright-blue",name:"Ultralytics Bright Blue",'
        f'colors:[{colors}],inactive:{{name:"Gray",lightHex:"#e0e0e0",darkHex:"#3b3b3b"}}}}'
    )


def _patch_index_js(text: str) -> str:
    """调整 TensorBoard 前端默认 smoothing、自动刷新和曲线颜色。"""
    text = _replace_once(text, "scalarSmoothing:.6", f"scalarSmoothing:{_SCALAR_SMOOTHING}")
    text = _replace_once(
        text,
        "reloadPeriodInMs:3e4,reloadEnabled:!1",
        f"reloadPeriodInMs:{_RELOAD_PERIOD_MS},reloadEnabled:!0",
    )
    text, count = _PALETTE_PATTERN.subn(_bright_blue_palette_js(), text, count=1)
    if count != 1:
        raise RuntimeError("TensorBoard 前端资源匹配失败：未找到默认调色板")
    return text


def _patch_index_html(text: str) -> str:
    """在页面启动前写入 TensorBoard 本地偏好，覆盖浏览器旧设置。"""
    script = (
        "<script>"
        "try{"
        'const k="_tb_global_settings";'
        "const v=JSON.parse(window.localStorage.getItem(k)||'{}');"
        f"v.scalarSmoothing={_SCALAR_SMOOTHING};"
        "v.autoReload=true;"
        f"v.autoReloadPeriodInMs={_RELOAD_PERIOD_MS};"
        "window.localStorage.setItem(k,JSON.stringify(v));"
        'window.localStorage.removeItem("_tb_global_settings.timeseries");'
        "}catch(e){}"
        "</script>"
    )
    return _replace_once(text, "<tb-webapp></tb-webapp>", f"{script}<tb-webapp></tb-webapp>")


@functools.lru_cache(maxsize=1)
def _patched_assets_zip_path() -> str:
    """复制并 patch TensorBoard webfiles.zip，返回临时 zip 路径。"""
    default_provider = assets.get_default_assets_zip_provider()
    if default_provider is None:
        raise RuntimeError("未找到 TensorBoard webfiles.zip")

    with default_provider() as source_file:
        with zipfile.ZipFile(source_file) as source_zip:
            target = tempfile.NamedTemporaryFile(
                prefix="ultralytics-tensorboard-webfiles-", suffix=".zip", delete=False
            )
            target_path = target.name
            target.close()

            seen_index_js = seen_index_html = False
            with zipfile.ZipFile(target_path, "w") as target_zip:
                for info in source_zip.infolist():
                    data = source_zip.read(info.filename)
                    if info.filename == "index.js":
                        data = _patch_index_js(data.decode("utf-8")).encode("utf-8")
                        seen_index_js = True
                    elif info.filename == "index.html":
                        data = _patch_index_html(data.decode("utf-8")).encode("utf-8")
                        seen_index_html = True
                    target_zip.writestr(info, data)

    if not seen_index_js or not seen_index_html:
        with contextlib.suppress(OSError):
            Path(target_path).unlink()
        missing = "index.js" if not seen_index_js else "index.html"
        raise RuntimeError(f"TensorBoard webfiles.zip 缺少 {missing}")
    return target_path


def get_assets_zip_provider():
    """返回定制后的 TensorBoard 前端资源 provider。"""
    try:
        path = _patched_assets_zip_path()
    except Exception as e:
        print(f"TensorBoard 自定义界面资源不可用，将使用默认资源：{e}", file=sys.stderr)
        return assets.get_default_assets_zip_provider()
    return lambda: open(path, "rb")


def main(argv: list[str] | None = None) -> int:
    """TensorBoard CLI 入口，保持原生参数兼容。"""
    argv = argv if argv is not None else sys.argv
    tensorboard = program.TensorBoard(plugins=default.get_plugins(), assets_zip_provider=get_assets_zip_provider())
    tensorboard.configure(argv)
    return tensorboard.main()


if __name__ == "__main__":
    raise SystemExit(main())
