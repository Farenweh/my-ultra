# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ultralytics.utils.plotting import plot_results


@pytest.mark.parametrize(
    ("loss_names", "expected_loss_files"),
    [
        (("box_loss", "cls_loss", "dfl_loss"), {"loss_box.png", "loss_cls.png", "loss_dfl.png"}),
        (("giou_loss", "cls_loss", "l1_loss"), {"loss_giou.png", "loss_cls.png", "loss_l1.png"}),
    ],
)
def test_plot_results_writes_overview_and_split_plots(tmp_path, monkeypatch, loss_names, expected_loss_files):
    """Verify plot_results preserves results.png and writes high-resolution split plots."""
    axvline_calls, scatter_calls, annotate_calls, legend_calls = [], [], [], []
    axvline, scatter, annotate, legend = Axes.axvline, Axes.scatter, Axes.annotate, Axes.legend

    def axvline_spy(self, x=0, *args, **kwargs):
        axvline_calls.append((x, kwargs.copy()))
        return axvline(self, x, *args, **kwargs)

    def scatter_spy(self, *args, **kwargs):
        scatter_calls.append((args, kwargs.copy()))
        return scatter(self, *args, **kwargs)

    def annotate_spy(self, text, *args, **kwargs):
        annotate_calls.append(text)
        return annotate(self, text, *args, **kwargs)

    def legend_spy(self, *args, **kwargs):
        legend_calls.append({"loc": kwargs.get("loc"), "title": self.get_title()})
        return legend(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "axvline", axvline_spy)
    monkeypatch.setattr(Axes, "scatter", scatter_spy)
    monkeypatch.setattr(Axes, "annotate", annotate_spy)
    monkeypatch.setattr(Axes, "legend", legend_spy)
    monkeypatch.setattr(Figure, "savefig", lambda _, fname, *args, **kwargs: fname.write_bytes(b"plot"))

    metric_names = (
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    )
    columns = (
        ("epoch", "time")
        + tuple(f"train/{name}" for name in loss_names)
        + metric_names
        + tuple(f"val/{name}" for name in loss_names)
        + ("fitness", "lr/pg0")
    )
    rows = [
        (1, 10, 1.0, 2.0, 3.0, 0.10, 0.20, 0.30, 0.40, 1.5, 2.5, 3.5, 0.10, 0.001),
        (2, 20, 0.8, 1.8, 2.8, 0.15, 0.25, 0.35, 0.45, "nan", "nan", "nan", 0.90, 0.002),
        (3, 30, 0.6, 1.6, 2.6, 0.20, 0.30, 0.40, 0.50, 1.1, 2.1, 3.1, 0.20, 0.003),
    ]
    results_csv = tmp_path / "results.csv"
    results_csv.write_text(
        "\n".join([",".join(columns), *(",".join(map(str, row)) for row in rows)]) + "\n",
        encoding="utf-8",
    )

    plotted = []
    plot_results(file=results_csv, on_plot=plotted.append)

    split_dir = tmp_path / "results"
    expected_files = expected_loss_files | {
        "loss_total.png",
        "metric_precision_box.png",
        "metric_recall_box.png",
        "metric_map50_box.png",
        "metric_map50_95_box.png",
    }
    expected_paths = {split_dir / name for name in expected_files}

    assert (tmp_path / "results.png").is_file()
    assert split_dir.is_dir()
    assert expected_paths == {p for p in split_dir.iterdir() if p.suffix == ".png"}
    assert (tmp_path / "results.png") in plotted
    assert expected_paths.issubset(set(plotted))
    assert all(path.stat().st_size > 0 for path in expected_paths)
    assert {2.0, 3.0}.issubset({float(x) for x, _ in axvline_calls})
    assert scatter_calls
    assert any(text.startswith("best train ") for text in annotate_calls)
    assert any(text.startswith("last val ") for text in annotate_calls)
    assert legend_calls
    legend_locs = {call["title"]: call["loc"] for call in legend_calls}
    assert legend_locs.get("train/cls_loss") == "upper right"  # results.png overview plot stays unchanged
    for name in loss_names:
        assert legend_locs.get(f"{name} train vs val") == "upper right"
    assert legend_locs.get("total_loss train vs val") == "upper right"
    for name in metric_names:
        assert legend_locs.get(name) == "lower right"


@pytest.mark.parametrize(
    ("metric_columns", "metric_rows"),
    [
        (("metrics/mAP50-95(B)",), ((0.4,), (0.8,), (0.5,))),
        (("metrics/mAP50-95(B)", "metrics/mAP50-95(M)"), ((0.9, 0.0), (0.5, 0.5), (0.4, 0.4))),
        (("metrics/mAP50-95(B)", "metrics/mAP50-95(P)"), ((0.9, 0.0), (0.5, 0.5), (0.4, 0.4))),
        (("metrics/accuracy_top1", "metrics/accuracy_top5"), ((0.9, 0.1), (0.6, 0.6), (0.5, 0.5))),
    ],
)
def test_plot_results_legacy_best_fitness_fallback(tmp_path, monkeypatch, metric_columns, metric_rows):
    """Verify legacy CSV files without fitness infer best rows from task-specific metrics."""
    axvline_calls = []
    axvline = Axes.axvline

    def axvline_spy(self, x=0, *args, **kwargs):
        axvline_calls.append((x, kwargs.copy()))
        return axvline(self, x, *args, **kwargs)

    monkeypatch.setattr(Axes, "axvline", axvline_spy)
    monkeypatch.setattr(Figure, "savefig", lambda _, fname, *args, **kwargs: fname.write_bytes(b"plot"))

    columns = ("epoch", "time", "train/loss", *metric_columns, "val/loss", "lr/pg0")
    rows = [
        (1, 10, 1.0, *metric_rows[0], 1.2, 0.001),
        (2, 20, 0.8, *metric_rows[1], 1.0, 0.002),
        (3, 30, 0.6, *metric_rows[2], 0.9, 0.003),
    ]
    results_csv = tmp_path / "results.csv"
    results_csv.write_text(
        "\n".join([",".join(columns), *(",".join(map(str, row)) for row in rows)]) + "\n",
        encoding="utf-8",
    )

    plot_results(file=results_csv)

    assert {2.0, 3.0}.issubset({float(x) for x, _ in axvline_calls})
