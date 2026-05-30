"""Phase 2b — episode-split linear probe + confusion matrix + checkpoint timeline.

Ported from osl_analysis ``phase2b_probe.py`` to the 2D label set.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Analysis.osl2d._io import analysis_dir, load_traces, write_json
from Analysis.osl2d.probe import probe_classification_episode_split
from Analysis.osl2d.segment import LABELS, INT_TO_LABEL, N_LABELS


def run(run_dir: Path, ckpt_labels=None, seed: int = 0, train_episodes_now: int = None):
    traces = load_traces(run_dir, ckpt_labels)
    out_dir = analysis_dir(run_dir)
    if traces.h.shape[1] == 0:
        write_json(out_dir / "phase2b_probe.json", {"skipped": "no recurrent hidden state."})
        return

    overall = probe_classification_episode_split(
        traces.h, traces.label.astype(int), traces.episode_id,
        test_size=0.2, seed=seed, return_weights=False,
    )

    per_ckpt = {}
    ckpts = sorted(set(traces.ckpt_label.tolist()))
    if len(ckpts) > 1:
        for cl in ckpts:
            m = (traces.ckpt_label == cl)
            if m.sum() < 50:
                continue
            per_ckpt[str(cl)] = probe_classification_episode_split(
                traces.h[m], traces.label[m].astype(int), traces.episode_id[m],
                test_size=0.2, seed=seed,
            )

    write_json(out_dir / "phase2b_probe.json", {
        "overall": overall,
        "per_checkpoint": per_ckpt,
        "label_order": list(LABELS),
    })

    if overall.get("confusion"):
        cm = np.asarray(overall["confusion"], dtype=float)
        classes = overall.get("classes", list(range(N_LABELS)))
        names = [INT_TO_LABEL.get(c, str(c)) for c in classes]
        row_sum = cm.sum(axis=1, keepdims=True).clip(min=1)
        cmn = cm / row_sum
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"Probe confusion  acc={overall['acc']:.3f}")
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                        color="black" if cmn[i, j] < 0.5 else "white", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / "phase2b_confusion.png", dpi=150)
        plt.close(fig)

    if train_episodes_now is not None and overall.get("acc") is not None:
        timeline_path = out_dir / "phase2b_timeline.json"
        rows = []
        if timeline_path.exists():
            try:
                rows = json.loads(timeline_path.read_text()).get("rows", [])
            except Exception:
                rows = []
        rows.append({
            "ts": time.time(),
            "train_episodes": int(train_episodes_now),
            "ckpt_labels": ckpts,
            "acc": overall.get("acc"),
            "macro_f1": overall.get("macro_f1"),
            "per_label_f1": overall.get("per_label_f1", []),
        })
        timeline_path.write_text(json.dumps({"rows": rows}, indent=2))

        xs = [r["train_episodes"] for r in rows]
        ys = [r["acc"] for r in rows if r["acc"] is not None]
        if len(ys) >= 1:
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.plot(xs, ys, marker="o", color="steelblue", label="probe acc")
            ax.set_xlabel("train episodes"); ax.set_ylabel("probe accuracy")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            ax.set_title("Probe accuracy vs. training progress")
            fig.tight_layout()
            fig.savefig(out_dir / "phase2b_timeline.png", dpi=150)
            plt.close(fig)


def build_parser():
    p = argparse.ArgumentParser("Analysis.osl2d.phase2b_probe")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ckpt-labels", nargs="*", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-episodes-now", type=int, default=None)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    run(Path(a.run_dir), a.ckpt_labels, a.seed, a.train_episodes_now)


if __name__ == "__main__":
    main()
