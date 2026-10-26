"""Tables and figures from the experiment stages (see experiments.py).

    from rdt_xai.report import build_report
    build_report("/content/drive/MyDrive/rdt-xai/run1")   # writes <out>/report/*
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABEL = dict(random="Random", daam="DAAM (attention)", gae="GAE (grad x attn)", grad="Gradient",
             gxi="Grad x Input", occlusion="Occlusion", iss="ISS (ICML'26)", ig_zero="IG (zero)",
             ig_bg="IG (bg)", blurig="BlurIG (orig. repo)", eig="Expected IG (ours)")


def _ci(x, n_boot=2000, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, (n_boot, len(x))).mean(1)
    return x.mean(), *np.percentile(bs, [2.5, 97.5])


def _fmt(m, lo, hi):
    return "n/a" if not np.isfinite(m) else f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def _wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def load_benchmark(out):
    rows = []
    for f in glob.glob(os.path.join(out, "B", "*.json")):
        r = json.load(open(f))
        for m, v in r["methods"].items():
            rows.append(dict(moment=r["moment"], task=r["task"], call=r["call"], step=r["step"],
                             phase=r["call"] / max(1, r["n_calls"] - 1), method=m,
                             **{k: v[k] for k in v if not k.startswith("curve") and np.isscalar(v[k])}))
    return pd.DataFrame(rows)


def benchmark_tables(df, ref="eig"):
    metrics = [c for c in ("deletion_morf", "insertion", "deletion_lerf", "morf_minus_lerf", "seed_spearman",
                           "seed_top10_iou", "completeness_rel_err", "prev_share", "seconds") if c in df]
    rows = []
    for m, g in df.groupby("method"):
        row = {"method": LABEL.get(m, m), "key": m, "n": len(g)}
        for c in metrics:
            row[c] = _fmt(*_ci(g[c]))
            row[c + "_mean"] = g[c].mean()
        rows.append(row)
    t = pd.DataFrame(rows).sort_values("deletion_morf_mean", ascending=False)
    # paired tests: reference method vs every other, per moment
    tests = []
    piv = {c: df.pivot_table(index="moment", columns="method", values=c) for c in ("deletion_morf", "insertion")}
    for m in df["method"].unique():
        if m == ref or ref not in df["method"].unique():
            continue
        row = {"vs": LABEL.get(m, m)}
        for c, P in piv.items():
            d = (P[ref] - P[m]).dropna()
            row[c + "_diff"] = d.mean()
            row[c + "_winrate"] = (d > 0).mean()
            row[c + "_p"] = wilcoxon(d).pvalue if len(d) > 5 and np.any(d != 0) else np.nan
        tests.append(row)
    return t, pd.DataFrame(tests)


def plot_curves(out, figdir):
    curves = {}
    for f in glob.glob(os.path.join(out, "B", "*.json")):
        r = json.load(open(f))
        for m, v in r["methods"].items():
            for cv in ("curve_morf", "curve_ins"):
                if cv in v:
                    curves.setdefault((m, cv), []).append(v[cv])
    if not curves:
        return
    from .metrics import FRACS_FAST, FRACS_FULL
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, cv, title in ((axes[0], "curve_morf", "Deletion (most relevant first)  - higher is better"),
                          (axes[1], "curve_ins", "Insertion (most relevant first)  - higher is better")):
        for (m, c), ys in sorted(curves.items()):
            if c != cv:
                continue
            Y = np.array(ys)
            x = np.array(FRACS_FULL if Y.shape[1] == len(FRACS_FULL) else FRACS_FAST)
            ax.plot(x, Y.mean(0), "-o", ms=3, lw=2.5 if m == "eig" else 1.2, label=LABEL.get(m, m),
                    color="k" if m == "random" else None, ls="--" if m == "random" else "-")
        ax.set_xlabel("fraction of patches removed" if cv == "curve_morf" else "fraction of patches revealed")
        ax.set_ylabel("plan deviation / full-removal deviation" if cv == "curve_morf" else "1 - normalized deviation")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=.3)
    axes[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_faithfulness_curves.png"), dpi=150)
    plt.close(fig)


def plot_qualitative(out, rollout_dir, figdir, methods=("daam", "gae", "iss", "blurig", "eig"), n=6):
    from PIL import Image
    files = sorted(glob.glob(os.path.join(out, "B", "*.npz")))
    if not files:
        return
    pick = [files[int(i)] for i in np.linspace(0, len(files) - 1, min(n, len(files)))]
    methods = [m for m in methods if m in np.load(pick[0]).files]
    fig, axes = plt.subplots(len(pick), len(methods) + 1, figsize=(2.1 * (len(methods) + 1), 2.1 * len(pick)))
    axes = np.atleast_2d(axes)
    for r, f in enumerate(pick):
        tag = os.path.basename(f)[:-4]
        ep, call = tag.rsplit("_call", 1)
        img = Image.open(os.path.join(rollout_dir, ep, f"call{int(call):03d}_curr.png")).convert("RGB").resize((384, 384))
        z = np.load(f)
        axes[r, 0].imshow(img)
        axes[r, 0].set_ylabel(f"{ep.split('_')[0][:-3]}\ncall {int(call)}", fontsize=7)
        for c, m in enumerate(methods, 1):
            a = np.clip(z[m], 0, None) if m not in ("daam", "gae", "iss", "grad") else z[m]
            a = (a - a.min()) / (a.max() - a.min() + 1e-12)
            up = np.kron(a, np.ones((14, 14)))
            up = np.pad(up, ((0, 6), (0, 6)))
            axes[r, c].imshow(img)
            axes[r, c].imshow(up, cmap="inferno", alpha=np.sqrt(up) * 0.85, vmin=0, vmax=1)
            if r == 0:
                axes[r, c].set_title(LABEL.get(m, m), fontsize=7)
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_qualitative.png"), dpi=150)
    plt.close(fig)


def denoise_report(out, figdir):
    recs = [json.load(open(f)) for f in glob.glob(os.path.join(out, "C", "*.json"))]
    if not recs:
        return None
    mass, drop, only = (np.array([r[k] for r in recs]) for k in ("step_mass", "knock_drop", "knock_only"))
    T = mass.shape[1]
    lab = [f"step {i+1}\n(noisiest)" if i == 0 else f"step {i+1}" for i in range(T)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, Y, title in ((axes[0], mass, "Share of |expected IG| by denoising step"),
                         (axes[1], drop, "Plan change when vision removed at one step"),
                         (axes[2], only, "Plan recovered when vision given at one step only")):
        m = Y.mean(0)
        ci = np.array([_ci(Y[:, i])[1:] for i in range(T)]).T
        ax.bar(range(T), m, yerr=np.abs(ci - m), capsize=3, color="#4C72B0")
        ax.set_xticks(range(T), lab, fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_denoising_steps.png"), dpi=150)
    plt.close(fig)
    # spatial maps per step for a few moments
    files = sorted(glob.glob(os.path.join(out, "C", "*.npz")))[:4]
    if files:
        fig, axes = plt.subplots(len(files), T, figsize=(2 * T, 2 * len(files)))
        axes = np.atleast_2d(axes)
        for r, f in enumerate(files):
            S = np.load(f)["steps"]
            v = np.abs(S).max()
            for c in range(T):
                axes[r, c].imshow(S[c], cmap="RdBu_r", vmin=-v, vmax=v)
                axes[r, c].set_xticks([]), axes[r, c].set_yticks([])
                if r == 0:
                    axes[r, c].set_title(lab[c].replace("\n", " "), fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "fig_denoising_step_maps.png"), dpi=150)
        plt.close(fig)
    return pd.DataFrame(dict(step=range(1, T + 1), attribution_share=mass.mean(0),
                             drop_one=drop.mean(0), only_one=only.mean(0), n=len(recs)))


def closed_loop_report(out, figdir):
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(out, "D", "*.json"))]
    if not rows:
        return None, None
    df = pd.DataFrame(rows)
    df["condition"] = df["method"] + ":" + df["mode"]
    tab = []
    for (task, cond), g in df.groupby(["task", "condition"]):
        p, lo, hi = _wilson(g.success.sum(), len(g))
        tab.append(dict(task=task, condition=cond, n=len(g), success=p, ci_lo=lo, ci_hi=hi))
    tab = pd.DataFrame(tab)
    # exact paired sign test on discordant episodes: eig:top vs random:random
    tests = []
    for task, g in df.groupby("task"):
        P = g.pivot_table(index="episode", columns="condition", values="success", aggfunc="first")
        for a, b in (("random:random", "eig:top"), ("eig:bottom", "eig:top"), ("random:random", "daam:top")):
            if a in P and b in P:
                x = P[[a, b]].dropna().astype(bool)
                n10, n01 = int((x[a] & ~x[b]).sum()), int((~x[a] & x[b]).sum())
                pv = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else np.nan
                tests.append(dict(task=task, a=a, b=b, a_only_success=n10, b_only_success=n01, p_sign_test=pv))
    tasks = sorted(tab.task.unique())
    conds = sorted(tab.condition.unique(), key=lambda c: (c != "none:none", c))
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.5 * len(tasks), 3.6), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        t = tab[tab.task == task].set_index("condition").reindex(conds)
        ax.bar(range(len(conds)), t.success, yerr=[t.success - t.ci_lo, t.ci_hi - t.success], capsize=3,
               color=["#999" if c.startswith(("none", "random")) else "#C44E52" if c.endswith("top") else "#55A868"
                      for c in conds])
        ax.set_xticks(range(len(conds)), [c.replace(":", "\n") for c in conds], fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_title(f"{task}: success with 10% of patches hidden", fontsize=9)
        ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_closed_loop.png"), dpi=150)
    plt.close(fig)
    return tab, pd.DataFrame(tests)


def sanity_report(out, figdir):
    f = os.path.join(out, "E", "sanity.json")
    if not os.path.exists(f):
        return None
    df = pd.DataFrame(json.load(open(f)))
    t = df.groupby(["method", "randomized_blocks"]).spearman.mean().unstack()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for m, row in t.iterrows():
        ax.plot([0] + list(row.index), [1] + list(row.values), "-o", label=LABEL.get(m, m))
    ax.set_xlabel("RDT blocks re-initialized (from the output side)")
    ax.set_ylabel("Spearman vs trained-model map")
    ax.set_title("Model-randomization sanity check (lower = method depends on the model)", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_sanity.png"), dpi=150)
    plt.close(fig)
    return t


def language_report(out, figdir):
    from .policy import JOINT_NAMES
    recs = [json.load(open(f)) for f in glob.glob(os.path.join(out, "F", "*.json"))]
    if not recs:
        return None, None
    tasks = sorted({r["task"] for r in recs})
    order = list(range(7, -1, -1))                                       # gripper on top, as in the repo's figures
    fig, axes = plt.subplots(len(tasks), 2, figsize=(14, 3.2 * len(tasks)), squeeze=False)
    for row, task in zip(axes, tasks):
        rs = [r for r in recs if r["task"] == task]
        words = rs[0]["words"]
        for ax, key, title in ((row[0], "word_drop", "word removal (mask)"), (row[1], "word_ig", "|IG| (zero baseline)")):
            M = np.mean([np.abs(np.array(r[key])) for r in rs], 0)
            M = M / (M.max(1, keepdims=True) + 1e-12)
            ax.imshow(M[order], cmap="inferno", aspect="auto", vmin=0, vmax=1)
            ax.set_xticks(range(len(words)), words, rotation=60, ha="right", fontsize=7)
            ax.set_yticks(range(8), [JOINT_NAMES[j] for j in order], fontsize=7)
            ax.set_title(f"{task}: {title}, row-normalized, n={len(rs)}", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_word_joint.png"), dpi=150)
    plt.close(fig)

    df = pd.DataFrame([dict(task=r["task"], phase=r["call"] / max(1, r["n_calls"] - 1), **r["modality"],
                            ig_drop_spearman=np.nanmean(r["ig_drop_spearman"]),
                            ig_completeness=np.mean(r["ig_completeness_rel_err"])) for r in recs])
    tot = df[["vision", "language", "state"]].abs().sum(1) + 1e-12
    for k in ("vision", "language", "state"):
        df[k + "_share"] = df[k].abs() / tot
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.5 * len(tasks), 3.3), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        g = df[df.task == task].sort_values("phase")
        bins = np.clip((g.phase * 4).astype(int), 0, 3)
        m = g.groupby(bins)[["vision_share", "language_share", "state_share"]].mean()
        ax.stackplot(m.index / 4 + 0.125, m.T.values, labels=["vision", "language", "proprioception"], alpha=.85)
        ax.set_xlim(0, 1), ax.set_ylim(0, 1)
        ax.set_xlabel("episode progress"), ax.set_title(f"{task}: modality reliance (Shapley share)", fontsize=8)
    axes[0][0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_modality.png"), dpi=150)
    plt.close(fig)
    summ = df.groupby("task")[["vision_share", "language_share", "state_share", "ig_drop_spearman", "ig_completeness"]].mean()
    return summ, df


def build_report(out, rollout_dir=None):
    rollout_dir = rollout_dir or os.path.join(out, "A")
    figdir = os.path.join(out, "report")
    os.makedirs(figdir, exist_ok=True)
    md = ["# RDT-1B attribution study - results\n"]

    roll = [json.load(open(f)) for f in glob.glob(os.path.join(rollout_dir, "*", "episode.json"))]
    if roll:
        r = pd.DataFrame([{k: x[k] for k in ("task", "episode", "success", "steps")} for x in roll])
        s = r.groupby("task").agg(episodes=("success", "size"), success_rate=("success", "mean"),
                                  mean_steps=("steps", "mean"))
        s.to_csv(os.path.join(figdir, "A_success.csv"))
        md += ["## A. Policy success (official protocol)\n", s.to_markdown(floatfmt=".2f"), "\n"]

    df = load_benchmark(out)
    if len(df):
        t, tests = benchmark_tables(df)
        cols = ["method", "n"] + [c for c in ("deletion_morf", "insertion", "morf_minus_lerf", "seed_spearman",
                                              "completeness_rel_err", "prev_share", "seconds") if c in t]
        t[cols].to_csv(os.path.join(figdir, "B_faithfulness.csv"), index=False)
        tests.to_csv(os.path.join(figdir, "B_paired_tests.csv"), index=False)
        md += ["## B. Faithfulness at matched compute (mean [95% bootstrap CI] over moments)\n",
               t[cols].to_markdown(index=False), "\n",
               "Paired Wilcoxon tests, Expected IG minus each method (positive = ours better):\n",
               tests.to_markdown(index=False, floatfmt=".3g"), "\n"]
        per_task = df.pivot_table(index="method", columns="task", values="deletion_morf").rename(index=LABEL)
        md += ["Deletion AUC by task:\n", per_task.to_markdown(floatfmt=".3f"), "\n"]
        plot_curves(out, figdir)
        plot_qualitative(out, rollout_dir, figdir)

    dn = denoise_report(out, figdir)
    if dn is not None:
        dn.to_csv(os.path.join(figdir, "C_denoising.csv"), index=False)
        md += ["## C. When during denoising does vision matter?\n", dn.to_markdown(index=False, floatfmt=".3f"), "\n"]

    cl, cl_tests = closed_loop_report(out, figdir)
    if cl is not None:
        cl.to_csv(os.path.join(figdir, "D_closed_loop.csv"), index=False)
        md += ["## D. Closed-loop: success with 10% of patches hidden online (Wilson 95% CI)\n",
               cl.to_markdown(index=False, floatfmt=".2f"), "\n", "Paired sign tests on discordant episodes:\n",
               cl_tests.to_markdown(index=False, floatfmt=".3g"), "\n"]

    sn = sanity_report(out, figdir)
    if sn is not None:
        md += ["## E. Model-randomization sanity check (Spearman to trained-model map)\n",
               sn.rename(index=LABEL).to_markdown(floatfmt=".2f"), "\n"]

    lg, _ = language_report(out, figdir)
    if lg is not None:
        lg.to_csv(os.path.join(figdir, "F_language.csv"))
        md += ["## F. Modality reliance (Shapley share) and word attribution consistency\n",
               lg.to_markdown(floatfmt=".3f"), "\n",
               "`ig_drop_spearman`: per-joint rank agreement between |word IG| and word-removal effect.\n"]

    text = "\n".join(md)
    with open(os.path.join(figdir, "RESULTS.md"), "w") as f:
        f.write(text)
    return text
