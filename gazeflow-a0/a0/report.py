"""Decision gate, a0_results.json, a0_summary.md, and required plots
(sections 33-38)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PASS_MEDIAN_DEG = 3.0
PASS_P95_DEG = 4.5
BORDERLINE_MEDIAN_DEG = 4.5
BORDERLINE_P95_DEG = 6.75

NO_GEOMETRY_STATUS = "UNKNOWN-NO-PHYSICAL-GEOMETRY"


def decide_status(median_deg: float | None, p95_deg: float | None) -> str:
    """Section 33-34 gate. Degrees are never fabricated (section 31), so if
    physical screen geometry was not supplied the gate cannot be evaluated
    and the run is reported as inconclusive rather than guessed."""
    if median_deg is None or p95_deg is None:
        return NO_GEOMETRY_STATUS
    if median_deg <= PASS_MEDIAN_DEG and p95_deg <= PASS_P95_DEG:
        return "PASS-CANDIDATE"
    if median_deg <= BORDERLINE_MEDIAN_DEG and p95_deg <= BORDERLINE_P95_DEG:
        return "BORDERLINE"
    return "FAIL"


def build_results_json(
    run_id: str,
    screen: dict,
    camera: dict,
    calibration: dict,
    validation_df: pd.DataFrame,
) -> dict:
    has_deg = validation_df["error_deg"].notna().any()
    has_mm = validation_df["error_mm"].notna().any()

    median_deg = float(validation_df["error_deg"].median()) if has_deg else None
    p95_deg = float(validation_df["error_deg"].quantile(0.95)) if has_deg else None
    status = decide_status(median_deg, p95_deg)

    return {
        "run_id": run_id,
        "status": status,
        "screen": screen,
        "camera": camera,
        "calibration": calibration,
        "validation": {
            "targets": int(len(validation_df)),
            "median_error_norm": float(validation_df["error_norm"].median()),
            "p95_error_norm": float(validation_df["error_norm"].quantile(0.95)),
            "median_error_px": float(validation_df["error_px"].median()),
            "p95_error_px": float(validation_df["error_px"].quantile(0.95)),
            "median_error_mm": float(validation_df["error_mm"].median()) if has_mm else None,
            "p95_error_mm": float(validation_df["error_mm"].quantile(0.95)) if has_mm else None,
            "median_error_deg": median_deg,
            "p95_error_deg": p95_deg,
        },
    }


def build_summary_md(results: dict, validation_df: pd.DataFrame, tracking: dict, observations: list[str]) -> str:
    v = results["validation"]
    c = results["calibration"]
    cam = results["camera"]
    scr = results["screen"]

    worst = validation_df.loc[validation_df["error_norm"].idxmax()]
    center_mask = validation_df["target_x_norm"].between(0.3, 0.7) & validation_df["target_y_norm"].between(0.3, 0.7)
    edge_mask = ~center_mask
    center_median = validation_df.loc[center_mask, "error_norm"].median() if center_mask.any() else float("nan")
    edge_median = validation_df.loc[edge_mask, "error_norm"].median() if edge_mask.any() else float("nan")

    deg_line = (
        f"- Median: {v['median_error_deg']:.2f} deg\n- P95: {v['p95_error_deg']:.2f} deg"
        if v["median_error_deg"] is not None
        else "- Degrees unavailable (no physical screen dimensions were supplied)"
    )

    recommendation = {
        "PASS-CANDIDATE": "Continue to the broader Stage A feasibility test with multiple participants.",
        "BORDERLINE": "Consider a stronger gaze model (e.g. a pretrained appearance-based model) before further product engineering.",
        "FAIL": "This approach is too far from the product requirement as measured. Consider repositioning to coarse attention zones, external eye-tracking hardware, or stopping.",
        NO_GEOMETRY_STATUS: "Re-run with --screen-width-mm/--screen-height-mm supplied to evaluate the PASS/BORDERLINE/FAIL gate.",
    }[results["status"]]

    lines = [
        "# GazeFlow A0 Result",
        "",
        "## Decision",
        results["status"],
        "",
        "## Setup",
        f"- Camera index: {cam['index']}",
        f"- Resolution: {cam['width']}x{cam['height']} (effective FPS: {cam['effective_fps']:.1f})",
        f"- Screen: {scr['width_px']}x{scr['height_px']} px"
        + (f", {scr['width_mm']:.0f}x{scr['height_mm']:.0f} mm" if scr.get("width_mm") else " (physical size unknown)"),
        f"- Viewing distance: {scr['viewing_distance_mm']} mm",
        f"- Run ID (date): {results['run_id']}",
        "",
        "## Accuracy",
        deg_line,
        f"- Median normalized error: {v['median_error_norm']:.4f}",
        f"- P95 normalized error: {v['p95_error_norm']:.4f}",
        f"- Center-region median error (norm): {center_median:.4f}",
        f"- Edge/corner-region median error (norm): {edge_median:.4f}",
        f"- Worst target: {worst['point_id']} (error_norm={worst['error_norm']:.4f})",
        "",
        "## Calibration",
        f"- Selected Ridge alpha: {c['selected_ridge_alpha']}",
        f"- Cross-validation median held-out error (norm): {c['cv_median_error_norm']:.4f}",
        f"- Cross-validation worst held-out error (norm): {c['cv_worst_error_norm']:.4f}",
        "",
        "## Tracking Quality",
        f"- Effective FPS: {cam['effective_fps']:.1f}",
        f"- Valid frame ratio: {tracking.get('valid_frame_ratio', float('nan')):.2%}",
        f"- Face detection failures: {tracking.get('face_detection_failures', 'n/a')}",
        f"- Blink-rejected frames: {tracking.get('blink_rejected_frames', 'n/a')}",
        "",
        "## Observations",
    ]
    lines += ([f"- {o}" for o in observations] if observations else ["- None recorded."])
    lines += ["", "## Recommendation", recommendation, ""]
    return "\n".join(lines)


def generate_plots(validation_df: pd.DataFrame, screen_width_px: int, screen_height_px: int, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # validation_scatter.png
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(validation_df["target_x_norm"], validation_df["target_y_norm"], c="tab:blue", label="target", s=40)
    ax.scatter(validation_df["predicted_x_norm"], validation_df["predicted_y_norm"], c="tab:red", label="predicted", s=40)
    for _, row in validation_df.iterrows():
        ax.plot(
            [row["target_x_norm"], row["predicted_x_norm"]],
            [row["target_y_norm"], row["predicted_y_norm"]],
            c="gray", linewidth=0.8, alpha=0.7,
        )
    ax.invert_yaxis()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(1.05, -0.05)
    ax.set_xlabel("x (normalized)")
    ax.set_ylabel("y (normalized)")
    ax.set_title("Validation: target vs predicted")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "validation_scatter.png", dpi=150)
    plt.close(fig)

    # error_by_target.png
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(validation_df["point_id"], validation_df["error_norm"], color="tab:orange")
    ax.set_xlabel("validation point")
    ax.set_ylabel("error (normalized)")
    ax.set_title("Error by target")
    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.tight_layout()
    fig.savefig(out_dir / "error_by_target.png", dpi=150)
    plt.close(fig)

    # error_distribution.png
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(validation_df["error_norm"], bins=10, color="tab:green", edgecolor="black")
    ax.axvline(validation_df["error_norm"].median(), color="black", linestyle="--", label="median")
    ax.set_xlabel("error (normalized)")
    ax.set_ylabel("count")
    ax.set_title("Error distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "error_distribution.png", dpi=150)
    plt.close(fig)

    # screen_error_map.png
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(
        validation_df["target_x_norm"], validation_df["target_y_norm"],
        c=validation_df["error_norm"], cmap="RdYlGn_r", s=200, edgecolor="black",
    )
    for _, row in validation_df.iterrows():
        ax.annotate(row["point_id"], (row["target_x_norm"], row["target_y_norm"]), fontsize=7, ha="center", va="center")
    ax.invert_yaxis()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(1.05, -0.05)
    ax.set_xlabel("x (normalized)")
    ax.set_ylabel("y (normalized)")
    ax.set_title("Screen error map")
    fig.colorbar(sc, ax=ax, label="error (normalized)")
    fig.tight_layout()
    fig.savefig(out_dir / "screen_error_map.png", dpi=150)
    plt.close(fig)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))
