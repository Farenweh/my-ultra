# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import contextlib
import subprocess
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests import SOURCE
from ultralytics import YOLO, download
from ultralytics.utils import ASSETS_URL, DATASETS_DIR, SETTINGS
from ultralytics.utils.checks import check_requirements


@pytest.mark.slow
def test_tensorboard():
    """Test training with TensorBoard logging enabled."""
    SETTINGS["tensorboard"] = True
    YOLO("yolo26n-cls.yaml").train(data="imagenet10", imgsz=32, epochs=3, plots=False, device="cpu")
    SETTINGS["tensorboard"] = False


class FakeSummaryWriter:
    """Minimal TensorBoard writer used to test callback behavior without tensorboard installed."""

    def __init__(self, log_dir=None):
        self.log_dir = log_dir
        self.scalars = []
        self.text = []
        self.hparams = []
        self.images = []
        self.flushed = False
        self.closed = False

    def add_scalar(self, tag, scalar_value, global_step=None):
        self.scalars.append((tag, scalar_value, global_step))

    def add_text(self, tag, text_string, global_step=None):
        self.text.append((tag, text_string, global_step))

    def add_hparams(self, hparam_dict, metric_dict, run_name=None):
        self.hparams.append((hparam_dict, metric_dict, run_name))

    def add_image(self, tag, img_tensor, global_step=None, dataformats="CHW"):
        self.images.append((tag, img_tensor, global_step, dataformats))

    def flush(self):
        self.flushed = True

    def close(self):
        self.closed = True


def _write_test_image(path: Path):
    """Create a tiny image file for TensorBoard image logging tests."""
    from PIL import Image

    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(path)


def _fake_tb_trainer(tmp_path: Path, image_path: Path):
    """Create a small trainer-like object for TensorBoard callback unit tests."""
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    best, last = weights_dir / "best.pt", weights_dir / "last.pt"
    best.touch()
    last.touch()
    (tmp_path / "args.yaml").write_text("epochs: 2\n", encoding="utf-8")
    (tmp_path / "results.csv").write_text("epoch,train/loss\n1,0.2\n", encoding="utf-8")

    args = SimpleNamespace(
        model="yolo26n-cls.yaml",
        data=tmp_path / "data.yaml",
        epochs=2,
        imgsz=32,
        batch=4,
        project=None,
        name="tb_test",
        augmentations=[{"name": "demo"}],
        optional=None,
    )
    validator = SimpleNamespace(speed={"inference": 2.5}, plots={image_path: {"timestamp": 2.0}})
    trainer = SimpleNamespace(
        args=args,
        save_dir=tmp_path,
        wdir=weights_dir,
        best=best,
        last=last,
        csv=tmp_path / "results.csv",
        epoch=0,
        epoch_time=1.25,
        train_time_start=time.time() - 3.0,
        fitness=0.42,
        best_fitness=0.5,
        metrics={"metrics/accuracy_top1": 0.8, "metrics/accuracy_top5": 0.95},
        lr={"lr/pg0": 0.01},
        tloss=0.2,
        data={"path": tmp_path, "train": "train", "val": "val", "nc": 2, "names": {0: "cat", 1: "dog"}},
        plots={image_path: {"timestamp": 1.0}},
        validator=validator,
    )
    trainer.label_loss_items = lambda loss, prefix="train": {f"{prefix}/loss": loss}
    return trainer


def test_tensorboard_callback_records_experiment_details(monkeypatch, tmp_path):
    """Verify enhanced TensorBoard callbacks record metrics, metadata, hparams, images, and close the writer."""
    from ultralytics.utils.callbacks import tensorboard as tb

    image_path = tmp_path / "results.png"
    _write_test_image(image_path)
    trainer = _fake_tb_trainer(tmp_path, image_path)
    writer = FakeSummaryWriter()

    monkeypatch.setattr(tb, "WRITER", writer)
    monkeypatch.setattr(tb, "_MODEL_INFO_LOGGED", False)
    monkeypatch.setattr(tb, "_PROCESSED_PLOTS", {})
    monkeypatch.setattr(
        tb.torch_utils,
        "model_info_for_loggers",
        lambda trainer: {"model/parameters": 123, "model/GFLOPs": 1.5, "model/speed_PyTorch(ms)": 2.5},
    )

    tb.on_train_epoch_end(trainer)
    scalar_tags = {tag for tag, _, _ in writer.scalars}
    assert {"train/loss", "lr/pg0"}.issubset(scalar_tags)

    tb.on_fit_epoch_end(trainer)
    scalar_tags = {tag for tag, _, _ in writer.scalars}
    assert {
        "metrics/accuracy_top1",
        "time/epoch",
        "time/elapsed",
        "fitness",
        "fitness/best",
        "model/parameters",
        "model/GFLOPs",
        "model/speed_PyTorch(ms)",
    }.issubset(scalar_tags)

    tb.on_train_end(trainer)
    text_tags = {tag for tag, _, _ in writer.text}
    assert {"run/args", "run/data", "run/artifacts"}.issubset(text_tags)
    assert writer.hparams
    hparams, metrics, run_name = writer.hparams[0]
    assert hparams["augmentations"].startswith("[{")
    assert hparams["optional"] == "None"
    assert metrics["fitness"] == 0.42
    assert run_name == "hparams"
    assert writer.images and writer.images[0][3] == "HWC"
    assert writer.flushed

    tb.teardown(trainer)
    assert writer.closed
    assert tb.WRITER is None


def test_tensorboard_graph_traces_model_copy(monkeypatch):
    """确认 graph tracing 不会改变训练模型的 mode、缓存或参数。"""
    from ultralytics.utils.callbacks import tensorboard as tb

    class CacheModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3)
            self.cache = None

        def forward(self, x):
            return self.linear(x.mean((2, 3)))

    class GraphWriter:
        def __init__(self):
            self.graphs = []

        def add_graph(self, graph, inputs):
            self.graphs.append((graph, inputs))

    model = CacheModel().train()
    weight = model.linear.weight.detach().clone()
    writer = GraphWriter()
    traced = []

    def fake_trace(copied_model, image, strict=False):
        traced.append(copied_model)
        copied_model.cache = torch.ones(1)
        copied_model.linear.weight.add_(1)
        return copied_model

    monkeypatch.setattr(tb, "WRITER", writer)
    monkeypatch.setattr(tb.torch.jit, "trace", fake_trace)
    trainer = SimpleNamespace(model=model, args=SimpleNamespace(imgsz=8), data={"channels": 3})

    tb._log_tensorboard_graph(trainer)

    assert traced and traced[0] is not model
    assert model.training
    assert model.cache is None
    torch.testing.assert_close(model.linear.weight, weight)
    assert writer.graphs


def test_tensorboard_graph_copy_failure_is_nonfatal(monkeypatch):
    """确认模型副本创建失败时只禁用 graph 记录，不中断训练。"""
    from ultralytics.utils.callbacks import tensorboard as tb

    warnings = []
    monkeypatch.setattr(tb, "deepcopy", lambda model: (_ for _ in ()).throw(MemoryError("copy failed")))
    monkeypatch.setattr(tb.LOGGER, "warning", warnings.append)
    trainer = SimpleNamespace(model=torch.nn.Linear(3, 3), args=SimpleNamespace(imgsz=8), data={"channels": 3})

    tb._log_tensorboard_graph(trainer)

    assert len(warnings) == 1
    assert "copy failed" in warnings[0]


class FakeTensorBoardProcess:
    """用于测试 TensorBoard 子进程生命周期的假进程。"""

    def __init__(self, command):
        self.command = command
        self.returncode = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def kill(self):
        self.killed = True
        self.returncode = -9


def _tb_environment(logdir: Path | str):
    """生成 TensorBoard /data/environment 风格响应。"""
    return {"data_location": str(logdir), "debug": {"flags": {"logdir": str(logdir)}}}


def _reset_tb_server_state(monkeypatch, tb):
    """重置 TensorBoard 服务相关全局状态。"""
    monkeypatch.setattr(tb, "_TENSORBOARD_PROCESS", None)
    monkeypatch.setattr(tb, "_TENSORBOARD_PORT", None)
    monkeypatch.setattr(tb, "_TENSORBOARD_LOGDIR", None)
    monkeypatch.setattr(tb, "WRITER", None)


def test_tensorboard_callback_starts_server_on_free_default_port(monkeypatch, tmp_path):
    """6006 空闲时应直接使用当前训练目录启动 TensorBoard。"""
    from ultralytics.utils.callbacks import tensorboard as tb

    _reset_tb_server_state(monkeypatch, tb)
    environments, commands = {}, []

    def fake_popen(command, stdout=None, stderr=None):
        commands.append(command)
        port = int(command[command.index("--port") + 1])
        logdir = command[command.index("--logdir") + 1]
        environments[port] = _tb_environment(logdir)
        return FakeTensorBoardProcess(command)

    monkeypatch.setattr(tb, "SummaryWriter", FakeSummaryWriter)
    monkeypatch.setattr(tb, "_tensorboard_environment", lambda port, timeout=1.0: environments.get(port))
    monkeypatch.setattr(tb, "_port_is_free", lambda port: port == 6006 and port not in environments)
    monkeypatch.setattr(tb.subprocess, "Popen", fake_popen)

    trainer = _fake_tb_trainer(tmp_path, tmp_path / "missing.png")
    tb.on_pretrain_routine_start(trainer)

    assert tb.WRITER.log_dir == str(tmp_path)
    assert commands[0][commands[0].index("-m") + 1] == "ultralytics.utils.tensorboard_launcher"
    assert commands and commands[0][commands[0].index("--logdir") + 1] == str(tmp_path.resolve())
    assert commands[0][commands[0].index("--port") + 1] == "6006"
    assert commands[0][commands[0].index("--reload_interval") + 1] == "5"
    assert commands[0][commands[0].index("--reload_multifile") + 1] == "true"
    assert tb._TENSORBOARD_PORT == 6006


def test_tensorboard_launcher_patches_frontend_defaults():
    """仓库内 TensorBoard launcher 应提供自动刷新、零 smoothing 和亮蓝曲线默认值。"""
    pytest.importorskip("tensorboard")
    from ultralytics.utils import tensorboard_launcher as tbl

    provider = tbl.get_assets_zip_provider()
    assert provider is not None

    with provider() as assets_zip:
        with zipfile.ZipFile(assets_zip) as zf:
            index_js = zf.read("index.js").decode("utf-8")
            index_html = zf.read("index.html").decode("utf-8")

    assert "scalarSmoothing:0" in index_js
    assert "reloadPeriodInMs:5000,reloadEnabled:!0" in index_js
    assert 'colorPalette:{id:"ultralytics-bright-blue"' in index_js
    assert "#009dff" in index_js
    assert "v.scalarSmoothing=0" in index_html
    assert "v.autoReload=true" in index_html
    assert "v.autoReloadPeriodInMs=5000" in index_html


def test_tensorboard_callback_reuses_matching_default_port(monkeypatch, tmp_path):
    """6006 已指向当前训练目录时不重复启动服务。"""
    from ultralytics.utils.callbacks import tensorboard as tb

    _reset_tb_server_state(monkeypatch, tb)
    monkeypatch.setattr(tb, "SummaryWriter", FakeSummaryWriter)
    monkeypatch.setattr(tb, "_tensorboard_environment", lambda port, timeout=1.0: _tb_environment(tmp_path))
    monkeypatch.setattr(tb.subprocess, "Popen", lambda *a, **k: pytest.fail("不应启动新的 TensorBoard 进程"))

    trainer = _fake_tb_trainer(tmp_path, tmp_path / "missing.png")
    tb.on_pretrain_routine_start(trainer)

    assert tb.WRITER.log_dir == str(tmp_path)
    assert tb._TENSORBOARD_PROCESS is None


def test_tensorboard_callback_restarts_wrong_default_port(monkeypatch, tmp_path):
    """6006 指向错误目录时应先尝试终止可见进程再重启 6006。"""
    from ultralytics.utils.callbacks import tensorboard as tb

    _reset_tb_server_state(monkeypatch, tb)
    environments = {6006: _tb_environment("/tmp")}
    terminated_ports, commands = [], []

    def fake_popen(command, stdout=None, stderr=None):
        commands.append(command)
        port = int(command[command.index("--port") + 1])
        logdir = command[command.index("--logdir") + 1]
        environments[port] = _tb_environment(logdir)
        return FakeTensorBoardProcess(command)

    monkeypatch.setattr(tb, "SummaryWriter", FakeSummaryWriter)
    monkeypatch.setattr(tb, "_tensorboard_environment", lambda port, timeout=1.0: environments.get(port))
    monkeypatch.setattr(tb, "_terminate_visible_tensorboard", lambda port: terminated_ports.append(port) or [1234])
    monkeypatch.setattr(tb, "_port_is_free", lambda port: port == 6006)
    monkeypatch.setattr(tb.subprocess, "Popen", fake_popen)

    trainer = _fake_tb_trainer(tmp_path, tmp_path / "missing.png")
    tb.on_pretrain_routine_start(trainer)

    assert terminated_ports == [6006]
    assert commands[0][commands[0].index("--port") + 1] == "6006"
    assert commands[0][commands[0].index("--logdir") + 1] == str(tmp_path.resolve())


def test_tensorboard_callback_falls_back_when_default_port_stays_busy(monkeypatch, tmp_path):
    """6006 无法释放时应使用后续空闲端口。"""
    from ultralytics.utils.callbacks import tensorboard as tb

    _reset_tb_server_state(monkeypatch, tb)
    environments = {6006: _tb_environment("/tmp")}
    commands = []

    def fake_popen(command, stdout=None, stderr=None):
        commands.append(command)
        port = int(command[command.index("--port") + 1])
        logdir = command[command.index("--logdir") + 1]
        environments[port] = _tb_environment(logdir)
        return FakeTensorBoardProcess(command)

    monkeypatch.setattr(tb, "SummaryWriter", FakeSummaryWriter)
    monkeypatch.setattr(tb, "_tensorboard_environment", lambda port, timeout=1.0: environments.get(port))
    monkeypatch.setattr(tb, "_terminate_visible_tensorboard", lambda port: [])
    monkeypatch.setattr(tb, "_port_is_free", lambda port: port == 6007)
    monkeypatch.setattr(tb.subprocess, "Popen", fake_popen)

    trainer = _fake_tb_trainer(tmp_path, tmp_path / "missing.png")
    tb.on_pretrain_routine_start(trainer)

    assert commands[0][commands[0].index("--port") + 1] == "6007"
    assert tb._TENSORBOARD_PORT == 6007


def test_tensorboard_callback_reuses_matching_fallback_port(monkeypatch, tmp_path):
    """6006 忙且 6007 已指向当前训练目录时应复用 6007。"""
    from ultralytics.utils.callbacks import tensorboard as tb

    _reset_tb_server_state(monkeypatch, tb)
    environments = {6007: _tb_environment(tmp_path)}
    terminated_ports = []

    monkeypatch.setattr(tb, "SummaryWriter", FakeSummaryWriter)
    monkeypatch.setattr(tb, "_tensorboard_environment", lambda port, timeout=1.0: environments.get(port))
    monkeypatch.setattr(tb, "_terminate_visible_tensorboard", lambda port: terminated_ports.append(port) or [])
    monkeypatch.setattr(tb, "_port_is_free", lambda port: False)
    monkeypatch.setattr(tb.subprocess, "Popen", lambda *a, **k: pytest.fail("不应启动新的 TensorBoard 进程"))

    trainer = _fake_tb_trainer(tmp_path, tmp_path / "missing.png")
    tb.on_pretrain_routine_start(trainer)

    assert tb.WRITER.log_dir == str(tmp_path)
    assert tb._TENSORBOARD_PROCESS is None
    assert terminated_ports == []


def test_tensorboard_callback_restarts_wrong_fallback_port(monkeypatch, tmp_path):
    """6007 指向错误目录且可释放时应在 6007 重启。"""
    from ultralytics.utils.callbacks import tensorboard as tb

    _reset_tb_server_state(monkeypatch, tb)
    environments = {6007: _tb_environment("/tmp/old-run")}
    terminated_ports, commands = [], []

    def fake_popen(command, stdout=None, stderr=None):
        commands.append(command)
        port = int(command[command.index("--port") + 1])
        logdir = command[command.index("--logdir") + 1]
        environments[port] = _tb_environment(logdir)
        return FakeTensorBoardProcess(command)

    monkeypatch.setattr(tb, "SummaryWriter", FakeSummaryWriter)
    monkeypatch.setattr(tb, "_tensorboard_environment", lambda port, timeout=1.0: environments.get(port))
    monkeypatch.setattr(tb, "_terminate_visible_tensorboard", lambda port: terminated_ports.append(port) or [4321])
    monkeypatch.setattr(tb, "_port_is_free", lambda port: port == 6007)
    monkeypatch.setattr(tb.subprocess, "Popen", fake_popen)

    trainer = _fake_tb_trainer(tmp_path, tmp_path / "missing.png")
    tb.on_pretrain_routine_start(trainer)

    assert terminated_ports == [6007]
    assert commands[0][commands[0].index("--port") + 1] == "6007"
    assert commands[0][commands[0].index("--logdir") + 1] == str(tmp_path.resolve())


def test_tensorboard_callback_skips_unreleased_wrong_fallback_port(monkeypatch, tmp_path):
    """6007 指向错误目录但无法释放时应继续尝试 6008。"""
    from ultralytics.utils.callbacks import tensorboard as tb

    _reset_tb_server_state(monkeypatch, tb)
    environments = {6007: _tb_environment("/tmp/old-run")}
    terminated_ports, commands = [], []

    def fake_popen(command, stdout=None, stderr=None):
        commands.append(command)
        port = int(command[command.index("--port") + 1])
        logdir = command[command.index("--logdir") + 1]
        environments[port] = _tb_environment(logdir)
        return FakeTensorBoardProcess(command)

    monkeypatch.setattr(tb, "SummaryWriter", FakeSummaryWriter)
    monkeypatch.setattr(tb, "_tensorboard_environment", lambda port, timeout=1.0: environments.get(port))
    monkeypatch.setattr(tb, "_terminate_visible_tensorboard", lambda port: terminated_ports.append(port) or [])
    monkeypatch.setattr(tb, "_port_is_free", lambda port: port == 6008)
    monkeypatch.setattr(tb.subprocess, "Popen", fake_popen)

    trainer = _fake_tb_trainer(tmp_path, tmp_path / "missing.png")
    tb.on_pretrain_routine_start(trainer)

    assert terminated_ports == [6007]
    assert commands[0][commands[0].index("--port") + 1] == "6008"


def test_tensorboard_callback_skips_non_tensorboard_busy_port(monkeypatch, tmp_path):
    """非 TensorBoard 服务占用端口时应跳过且不尝试终止。"""
    from ultralytics.utils.callbacks import tensorboard as tb

    _reset_tb_server_state(monkeypatch, tb)
    environments = {}
    terminated_ports, commands = [], []

    def fake_popen(command, stdout=None, stderr=None):
        commands.append(command)
        port = int(command[command.index("--port") + 1])
        logdir = command[command.index("--logdir") + 1]
        assert port == 6007
        environments[port] = _tb_environment(logdir)
        return FakeTensorBoardProcess(command)

    monkeypatch.setattr(tb, "SummaryWriter", FakeSummaryWriter)
    monkeypatch.setattr(tb, "_tensorboard_environment", lambda port, timeout=1.0: environments.get(port))
    monkeypatch.setattr(tb, "_terminate_visible_tensorboard", lambda port: terminated_ports.append(port) or [])
    monkeypatch.setattr(tb, "_port_is_free", lambda port: port == 6007 and port not in environments)
    monkeypatch.setattr(tb.subprocess, "Popen", fake_popen)

    trainer = _fake_tb_trainer(tmp_path, tmp_path / "missing.png")
    tb.on_pretrain_routine_start(trainer)

    assert terminated_ports == []
    assert commands[0][commands[0].index("--port") + 1] == "6007"


def test_tensorboard_callback_keeps_writer_when_server_start_fails(monkeypatch, tmp_path):
    """自动启动服务失败时不应影响 SummaryWriter 初始化。"""
    from ultralytics.utils.callbacks import tensorboard as tb

    _reset_tb_server_state(monkeypatch, tb)
    monkeypatch.setattr(tb, "SummaryWriter", FakeSummaryWriter)
    monkeypatch.setattr(tb, "_tensorboard_environment", lambda port, timeout=1.0: None)
    monkeypatch.setattr(tb, "_port_is_free", lambda port: port == 6006)
    monkeypatch.setattr(tb.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(OSError("启动失败")))

    trainer = _fake_tb_trainer(tmp_path, tmp_path / "missing.png")
    tb.on_pretrain_routine_start(trainer)

    assert tb.WRITER.log_dir == str(tmp_path)
    assert tb._TENSORBOARD_PROCESS is None


@pytest.mark.skipif(not check_requirements("ray", install=False), reason="ray[tune] not installed")
def test_model_ray_tune():
    """Tune YOLO model using Ray for hyperparameter optimization."""
    YOLO("yolo26n-cls.yaml").tune(
        use_ray=True, data="imagenet10", grace_period=1, iterations=1, imgsz=32, epochs=1, plots=False, device="cpu"
    )


@pytest.mark.skipif(not check_requirements("mlflow", install=False), reason="mlflow not installed")
def test_mlflow(tmp_path, monkeypatch):
    """Test training with MLflow tracking enabled."""
    import mlflow

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test_mlflow")
    monkeypatch.setitem(SETTINGS, "mlflow", True)
    try:
        YOLO("yolo26n-cls.yaml").train(data="imagenet10", imgsz=32, epochs=3, plots=False, device="cpu")
    finally:
        mlflow.autolog(disable=True)
        mlflow.end_run()


@pytest.mark.skipif(not check_requirements("mlflow", install=False), reason="mlflow not installed")
def test_mlflow_keep_run_active(tmp_path, monkeypatch):
    """Ensure MLFLOW_KEEP_RUN_ACTIVE controls whether new MLflow runs remain active."""
    import mlflow

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "keep_run_active")
    monkeypatch.setenv("MLFLOW_RUN", "Test Run")
    monkeypatch.setitem(SETTINGS, "mlflow", True)
    try:
        monkeypatch.setenv("MLFLOW_KEEP_RUN_ACTIVE", "True")
        YOLO("yolo26n-cls.yaml").train(data="imagenet10", imgsz=32, epochs=1, plots=False, device="cpu")
        active = mlflow.active_run()
        assert active is not None and active.info.status == "RUNNING", (
            "MLflow run should be active when MLFLOW_KEEP_RUN_ACTIVE=True"
        )
        mlflow.end_run()

        monkeypatch.setenv("MLFLOW_KEEP_RUN_ACTIVE", "False")
        YOLO("yolo26n-cls.yaml").train(data="imagenet10", imgsz=32, epochs=1, plots=False, device="cpu")
        assert mlflow.active_run() is None, "MLflow run should be ended when MLFLOW_KEEP_RUN_ACTIVE=False"

        monkeypatch.delenv("MLFLOW_KEEP_RUN_ACTIVE", raising=False)
        YOLO("yolo26n-cls.yaml").train(data="imagenet10", imgsz=32, epochs=1, plots=False, device="cpu")
        assert mlflow.active_run() is None, "MLflow run should be ended by default when MLFLOW_KEEP_RUN_ACTIVE is unset"
    finally:
        mlflow.autolog(disable=True)
        mlflow.end_run()


@pytest.mark.skipif(not check_requirements("tritonclient", install=False), reason="tritonclient[all] not installed")
def test_triton(tmp_path, isolated_model):
    """Test NVIDIA Triton Server functionalities with YOLO model."""
    check_requirements("tritonclient[all]")
    from tritonclient.http import InferenceServerClient

    # Create variables
    model_name = "yolo"
    triton_repo = tmp_path / "triton_repo"  # Triton repo path
    triton_model = triton_repo / model_name  # Triton model path

    # Export model to ONNX
    f = YOLO(isolated_model).export(format="onnx", dynamic=True)

    # Prepare Triton repo
    (triton_model / "1").mkdir(parents=True, exist_ok=True)
    Path(f).rename(triton_model / "1" / "model.onnx")
    (triton_model / "config.pbtxt").touch()

    # Define image https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver
    tag = "nvcr.io/nvidia/tritonserver:23.09-py3"  # 6.4 GB

    # Pull the image
    subprocess.call(f"docker pull {tag}", shell=True)

    # Run the Triton server and capture the container ID
    container_id = (
        subprocess.check_output(
            f"docker run -d --rm -v {triton_repo}:/models -p 8000:8000 {tag} tritonserver --model-repository=/models",
            shell=True,
        )
        .decode("utf-8")
        .strip()
    )

    # Wait for the Triton server to start
    triton_client = InferenceServerClient(url="localhost:8000", verbose=False, ssl=False)

    # Wait until model is ready
    for _ in range(10):
        with contextlib.suppress(Exception):
            assert triton_client.is_model_ready(model_name)
            break
        time.sleep(1)

    # Check Triton inference
    YOLO(f"http://localhost:8000/{model_name}", "detect")(SOURCE)  # exported model inference

    # Kill and remove the container at the end of the test
    subprocess.call(f"docker kill {container_id}", shell=True)


@pytest.mark.skipif(not check_requirements("faster-coco-eval", install=False), reason="faster-coco-eval not installed")
def test_faster_coco_eval():
    """Validate YOLO model predictions on COCO dataset using faster-coco-eval."""
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.models.yolo.pose import PoseValidator
    from ultralytics.models.yolo.segment import SegmentationValidator

    args = {"model": "yolo26n.pt", "data": "coco8.yaml", "save_json": True, "imgsz": 64}
    validator = DetectionValidator(args=args)
    validator()
    validator.is_coco = True
    download(f"{ASSETS_URL}/instances_val2017.json", dir=DATASETS_DIR / "coco8/annotations")
    _ = validator.eval_json(validator.stats)

    args = {"model": "yolo26n-seg.pt", "data": "coco8-seg.yaml", "save_json": True, "imgsz": 64}
    validator = SegmentationValidator(args=args)
    validator()
    validator.is_coco = True
    download(f"{ASSETS_URL}/instances_val2017.json", dir=DATASETS_DIR / "coco8-seg/annotations")
    _ = validator.eval_json(validator.stats)

    args = {"model": "yolo26n-pose.pt", "data": "coco8-pose.yaml", "save_json": True, "imgsz": 64}
    validator = PoseValidator(args=args)
    validator()
    validator.is_coco = True
    download(f"{ASSETS_URL}/person_keypoints_val2017.json", dir=DATASETS_DIR / "coco8-pose/annotations")
    _ = validator.eval_json(validator.stats)
