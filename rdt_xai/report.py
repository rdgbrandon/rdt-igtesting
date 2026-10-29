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
# the 8 executed ManiSkill joints in policy order (kept here so the report needs neither torch nor RDT)
JOINT_NAMES = ["base rot", "shoulder", "upper arm", "elbow", "forearm rot", "wrist pitch", "wrist rot", "gripper"]


def _ci(x, n_boot=2000, seed=0, stat=np.mean):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = stat(rng.choice(x, (n_boot, len(x))), axis=1)
    return stat(x), *np.percentile(bs, [2.5, 97.5])


def _fmt(m, lo, hi):
    return "n/a" if not np.isfinite(m) else f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def _wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, min(p, max(0.0, c - h)), max(p, min(1.0, c + h))      # rounding can put c - h above p = 0


def bench_dir(out):
    """Faithfulness scores to report: the inpainting re-score if present, else the pilot's grey fill."""
    d = os.path.join(out, "B_inpaint")
    return d if glob.glob(os.path.join(d, "*.json")) else os.path.join(out, "B")


def load_benchmark(bdir):
    rows = []
    for f in glob.glob(os.path.join(bdir, "*.json")):
        r = json.load(open(f))
        for m, v in r["methods"].items():
            rows.append(dict(moment=r["moment"], task=r["task"], call=r["call"], step=r["step"],
                             phase=r["call"] / max(1, r["n_calls"] - 1), method=m,
                             **{k: v[k] for k in v if not k.startswith("curve") and np.isscalar(v[k])}))
    return pd.DataFrame(rows)


FAITH = ("deletion_morf", "insertion")          # higher = better; scale shared by all methods within a moment
LOW_SIGNAL = 0.25                               # D_full below this fraction of the median D_full


def mean_rank(df, c):
    """Mean over moments of each method's rank on metric c within the moment (1 = best). Within a moment
    every method shares D_full, so ranks do not depend on the normalization."""
    return df.groupby("moment")[c].rank(ascending=False).groupby(df["method"]).mean()


def low_signal(df):
    """Moments where removing the whole view barely moves the plan: D/D_full divides by a small number
    there, so a method's mean can be dominated by one of them (pilot 3: ISS 16.2 on one moment)."""
    d = df.groupby("moment").d_full.first() if "d_full" in df else pd.Series(dtype=float)
    return d[d < LOW_SIGNAL * d.median()]


def benchmark_tables(df, ref="eig"):
    """Per method: median [95% bootstrap CI] and mean per-moment rank for deletion / insertion (robust
    to low-signal moments), mean [CI] for the rest; paired tests of `ref` against every other method."""
    metrics = [c for c in ("deletion_morf", "insertion", "deletion_lerf", "morf_minus_lerf", "seed_spearman",
                           "seed_top10_iou", "completeness_rel_err", "prev_share", "seconds") if c in df]
    ranks = {c: mean_rank(df, c) for c in FAITH}
    low = set(low_signal(df).index)
    rows = []
    for m, g in df.groupby("method"):
        row = {"method": LABEL.get(m, m), "key": m, "n": len(g)}
        for c in metrics:
            row[c] = _fmt(*_ci(g[c], stat=np.median if c in FAITH else np.mean))
            row[c + "_mean"] = g[c].mean()
        for c in FAITH:
            row[c + "_rank"] = ranks[c][m]
            row[c + "_median"] = g[c].median()
            row[c + "_mean_excl_low"] = g[~g.moment.isin(low)][c].mean()
        rows.append(row)
    t = pd.DataFrame(rows).sort_values("deletion_morf_rank")
    # paired tests: reference method vs every other, per moment
    tests = []
    piv = {c: df.pivot_table(index="moment", columns="method", values=c) for c in FAITH}
    for m in df["method"].unique():
        if m == ref or ref not in df["method"].unique():
            continue
        row = {"vs": LABEL.get(m, m)}
        for c, P in piv.items():
            d = (P[ref] - P[m]).dropna()
            row[c + "_median_diff"] = d.median()
            row[c + "_winrate"] = (d > 0).mean()
            row[c + "_p"] = wilcoxon(d).pvalue if len(d) > 5 and np.any(d != 0) else np.nan
        tests.append(row)
    return t, pd.DataFrame(tests)


def load_curves(bdir):
    curves = {}
    for f in glob.glob(os.path.join(bdir, "*.json")):
        r = json.load(open(f))
        for m, v in r["methods"].items():
            for cv in ("curve_morf", "curve_ins"):
                if cv in v:
                    curves.setdefault((m, cv), []).append(v[cv])
    return curves


def fill_comparison(out):
    """Pilot grey fill vs inpainting on the same maps and moments: median deletion, mean per-moment
    deletion rank, and `peak_morf`, the mean over moments of max D(f)/D_full along the deletion curve."""
    rows = {}
    for fill, sub in (("gray", "B"), ("inpaint", "B_inpaint")):
        df = load_benchmark(os.path.join(out, sub))
        if not len(df):
            return None
        rk = mean_rank(df, "deletion_morf")
        for m, g in df.groupby("method"):
            rows.setdefault(m, {"method": LABEL.get(m, m)})
            rows[m][f"deletion_median_{fill}"] = g.deletion_morf.median()
            rows[m][f"deletion_rank_{fill}"] = rk[m]
        for (m, cv), ys in load_curves(os.path.join(out, sub)).items():
            if cv == "curve_morf":
                rows[m][f"peak_morf_{fill}"] = np.mean([max(y) for y in ys])
    return pd.DataFrame(rows.values()).sort_values("deletion_rank_inpaint")


def plot_curves(bdir, figdir, name="fig_faithfulness_curves", note=""):
    curves = load_curves(bdir)
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
            ax.plot(x, Y.mean(0), marker="o", ms=3, lw=2.5 if m == "eig" else 1.2, label=LABEL.get(m, m),
                    color="k" if m == "random" else None, ls="--" if m == "random" else "-")
        ax.set_xlabel("fraction of patches removed" if cv == "curve_morf" else "fraction of patches revealed")
        ax.set_ylabel("plan deviation / full-removal deviation" if cv == "curve_morf" else "1 - normalized deviation")
        ax.set_title(title + note, fontsize=9)
        ax.grid(alpha=.3)
    axes[0].axhline(1, color="0.5", lw=0.8, ls=":")                       # = whole view removed
    axes[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, name + ".png"), dpi=150)
    plt.close(fig)


def plot_qualitative(bdir, rollout_dir, figdir, methods=("daam", "gae", "iss", "blurig", "eig"), n=6):
    from PIL import Image
    files = sorted(glob.glob(os.path.join(bdir, "*.npz")))
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
    """Reports D_inpaint (current protocol) if present, else the pilot's grey-fill D."""
    rows = []
    for sub in ("D_inpaint", "D"):
        rows = [json.load(open(f)) for f in glob.glob(os.path.join(out, sub, "*.json"))]
        if rows:
            break
    if not rows:
        return None, None, None
    df = pd.DataFrame(rows)
    if "fill" not in df:
        df["fill"] = "gray"
    desc = ", ".join(f"{100 * fr:g}% of patches hidden, {fl} fill"
                     for fr, fl in sorted(set(zip(df["frac"], df["fill"]))))
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
        for a, b in (("random:random", "eig:top_abs"), ("eig:bottom_abs", "eig:top_abs"),
                     ("random:random", "eig:top"), ("random:random", "daam:top"), ("random:random", "blurig:top"),
                     ("none:none", "eig:bottom_abs"), ("none:none", "random:random"), ("eig:bottom", "eig:top")):
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
               color=["#999" if c.startswith(("none", "random")) else "#C44E52" if ":top" in c else "#55A868"
                      for c in conds])
        ax.set_xticks(range(len(conds)), [c.replace(":", "\n") for c in conds], fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_title(f"{task}: success, {desc}", fontsize=8)
        ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_closed_loop.png"), dpi=150)
    plt.close(fig)
    return tab, pd.DataFrame(tests), desc


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

    df = _shares(pd.DataFrame([dict(task=r["task"], phase=r["call"] / max(1, r["n_calls"] - 1), **r["modality"],
                                    ig_drop_spearman=np.nanmean(r["ig_drop_spearman"]),
                                    ig_completeness=np.mean(r["ig_completeness_rel_err"])) for r in recs]))
    summ = df.groupby("task")[["vision_share", "language_share", "state_share", "ig_drop_spearman", "ig_completeness"]].mean()
    # reliance over the episode: every decision of whole episodes (F_modality) if present, else the F moments
    tl = [json.load(open(f)) for f in glob.glob(os.path.join(out, "F_modality", "*.json"))]
    if tl:
        df_t = _shares(pd.DataFrame([dict(task=e["task"], episode=e["episode"], phase=c["call"] / max(1, e["n_calls"] - 1),
                                          **{k: c[k] for k in ("vision", "language", "state")})
                                     for e in tl for c in e["calls"]]))
        timeline = df_t.groupby("task")[["vision_share", "language_share", "state_share"]].mean()
        timeline.insert(0, "episodes", df_t.groupby("task").episode.nunique())
        timeline.insert(1, "decisions", df_t.groupby("task").size())
    else:
        df_t, timeline = df, None
    nb = 8 if tl else 4
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.5 * len(tasks), 3.3), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        g = df_t[df_t.task == task]
        bins = np.clip((g.phase * nb).astype(int), 0, nb - 1)
        m = g.groupby(bins)[["vision_share", "language_share", "state_share"]].mean()
        x = (m.index.values + 0.5) / nb
        ax.stackplot(x, m.T.values, labels=["vision", "language", "proprioception"], alpha=.85)
        ax.plot(x, m.vision_share, "o", ms=3, color="k")                    # where the bins actually are
        ax.set_xlim(0, 1), ax.set_ylim(0, 1)
        ax.set_xlabel("episode progress")
        ax.set_title(f"{task}: modality reliance (Shapley share), {len(g)} decisions", fontsize=8)
    axes[0][0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_modality.png"), dpi=150)
    plt.close(fig)
    return summ, timeline


def _shares(df):
    tot = df[["vision", "language", "state"]].abs().sum(1) + 1e-12
    for k in ("vision", "language", "state"):
        df[k + "_share"] = df[k].abs() / tot
    return df


def grounding_report(out, figdir, show=("eig", "daam", "grad", "blurig", "random")):
    """Stage G: what each method claims the policy looks at (per scene object) vs what removing
    that object does to the plan."""
    recs = [json.load(open(f)) for f in glob.glob(os.path.join(out, "G", "*.json"))]
    recs = [r for r in recs if "effect" in r]
    if not recs:
        return None
    rows = []
    for r in recs:
        gs, area = r["groups"], np.array(r["area_share"])
        task_objs = [i for i, g in enumerate(gs) if g not in ("gripper", "arm", "background")]
        for m in r["claim"]:
            c = np.array(r["claim"][m])
            rows.append(dict(moment=r["moment"], task=r["task"], method=m, spearman=r["spearman"][m],
                             top1=r["top1"][m], on_object=1 - c[-1], task_objects=c[task_objs].sum(),
                             task_area=area[task_objs].sum()))
    df = pd.DataFrame(rows)
    methods = []
    for m, g in df.groupby("method"):
        methods.append(dict(method=LABEL.get(m, m), key=m, n=len(g), claim_vs_effect_spearman=_fmt(*_ci(g.spearman)),
                            spearman_mean=g.spearman.mean(), top1_agreement=g.top1.mean(),
                            task_object_share=g.task_objects.mean(),
                            task_object_enrichment=(g.task_objects / g.task_area.clip(1e-6)).median()))
    mt = pd.DataFrame(methods).sort_values("spearman_mean", ascending=False)

    per_obj, phase_rows = [], []
    for task in sorted({r["task"] for r in recs}):
        rs = [r for r in recs if r["task"] == task]
        gs = rs[0]["groups"]
        E = np.array([r["effect"] for r in rs], dtype=float)
        C = np.array([r["control"] for r in rs], dtype=float)
        A = np.array([r["area_share"] for r in rs])
        K = np.array([r["claim"]["eig"] for r in rs]) if "eig" in rs[0]["claim"] else None
        for i, g in enumerate(gs[:-1]):
            per_obj.append(dict(task=task, object=g, area_share=A[:, i].mean(), effect_median=np.nanmedian(E[:, i]),
                                control_median=np.nanmedian(C[:, i]),
                                effect_over_control=np.nanmedian(E[:, i] / np.where(C[:, i] > 0, C[:, i], np.nan)),
                                eig_claim=K[:, i].mean() if K is not None else np.nan))
        grasp = np.array([any(v > 0.5 for k, v in r.get("state", {}).items() if "grasped" in k) for r in rs])
        if grasp.any() and (~grasp).any():
            for ph, sel in (("not grasped", ~grasp), ("grasped", grasp)):
                for i, g in enumerate(gs[:-3]):
                    phase_rows.append(dict(task=task, phase=ph, n=int(sel.sum()), object=g,
                                           eig_claim=K[sel, i].mean() if K is not None else np.nan,
                                           effect_median=np.nanmedian(E[sel, i])))
    po, ph = pd.DataFrame(per_obj), pd.DataFrame(phase_rows)

    tasks = sorted(po.task.unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.6 * len(tasks), 3.4), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        rs = [r for r in recs if r["task"] == task]
        gs = rs[0]["groups"][:-1]
        E = np.nanmean(np.array([r["effect"] for r in rs], dtype=float)[:, :-1], 0)
        bars = {"plan change when removed": E / (np.nansum(E) + 1e-12)}
        for m in show:
            if m in rs[0]["claim"]:
                K = np.mean([r["claim"][m] for r in rs], 0)[:-1]
                bars[LABEL.get(m, m)] = K / (K.sum() + 1e-12)
        w = 0.8 / len(bars)
        for j, (lab, v) in enumerate(bars.items()):
            ax.bar(np.arange(len(gs)) + (j - (len(bars) - 1) / 2) * w, v, w, label=lab,
                   color="k" if j == 0 else None)
        ax.set_xticks(range(len(gs)), gs, fontsize=8)
        ax.set_title(f"{task}: share per object (excl. background), n={len(rs)}", fontsize=8)
        ax.grid(axis="y", alpha=.3)
    axes[0][0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_grounding.png"), dpi=150)
    plt.close(fig)
    seg = [json.load(open(f)) for f in glob.glob(os.path.join(out, "G", "seg", "*.json"))]
    replay = max((x["rgb_diff_max"] for x in seg), default=float("nan"))
    return mt, po, ph, replay


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

    bdir = bench_dir(out)
    fill = "inpainted" if bdir.endswith("_inpaint") else 'grey ("missing camera" colour)'
    df = load_benchmark(bdir)
    if len(df):
        t, tests = benchmark_tables(df)
        cols = ["method", "n", "deletion_morf", "deletion_morf_rank", "insertion", "insertion_rank"] + [
            c for c in ("morf_minus_lerf", "seed_spearman", "completeness_rel_err", "prev_share", "seconds") if c in t]
        t.drop(columns="key").to_csv(os.path.join(figdir, "B_faithfulness.csv"), index=False)
        tests.to_csv(os.path.join(figdir, "B_paired_tests.csv"), index=False)
        low = low_signal(df)
        low.rename("d_full").to_csv(os.path.join(figdir, "B_low_signal_moments.csv"))
        md += ["## B. Faithfulness at matched compute\n",
               f"Removed patches are {fill}. Scores are plan deviation divided by the deviation with the whole "
               "view removed (D_full). D_full is shared by every method within a moment, so per-moment "
               "comparisons (ranks, win rates, Wilcoxon) do not depend on it; scores above 1 are common, see "
               "the fill comparison below. `deletion_morf` / `insertion`: median [95% bootstrap CI] over "
               "moments; `*_rank`: mean per-moment rank (1 = best of 11); other columns: mean [95% CI].\n\n",
               t[cols].to_markdown(index=False, floatfmt=".2f"), "\n",
               "Paired Wilcoxon tests, Expected IG minus each method (positive = ours better; median "
               "per-moment difference):\n",
               tests.to_markdown(index=False, floatfmt=".3g"), "\n",
               f"Low-signal moments (D_full below {LOW_SIGNAL:g} x the median, where removing the whole view "
               f"barely moves the plan and ratios are unstable): {len(low)} of {df.moment.nunique()}"
               + (" - " + ", ".join(low.index) if len(low) else "") + ". Medians and ranks are insensitive to "
               "them; means with and without them:\n",
               t[["method", "deletion_morf_mean", "deletion_morf_mean_excl_low", "insertion_mean",
                  "insertion_mean_excl_low"]].to_markdown(index=False, floatfmt=".3f"), "\n"]
        per_task = df.pivot_table(index="method", columns="task", values="deletion_morf",
                                  aggfunc="median").rename(index=LABEL)
        md += ["Median deletion score by task:\n", per_task.to_markdown(floatfmt=".3f"), "\n"]
        plot_curves(bdir, figdir, note=f"  [{fill.split()[0]} fill]")
        plot_qualitative(os.path.join(out, "B"), rollout_dir, figdir)
        fc = fill_comparison(out) if bdir.endswith("_inpaint") else None
        if fc is not None:
            plot_curves(os.path.join(out, "B"), figdir, name="fig_faithfulness_curves_gray", note="  [grey fill]")
            fc.to_csv(os.path.join(figdir, "B_fill_comparison.csv"), index=False)
            md += ["Same maps, grey fill (pilot 2 metric) vs inpainting. `peak_morf` = mean over moments of the "
                   "maximum of D(f)/D_full along the deletion curve. It is ~2 under both fills, so it is not a "
                   "grey-square artifact: hiding a few top patches moves the plan more than hiding the camera "
                   "entirely (without a camera RDT falls back on joint state; with the object hidden it plans "
                   "for a different scene). The method ranking does change with the fill:\n",
                   fc.to_markdown(index=False, floatfmt=".2f"), "\n"]

    dn = denoise_report(out, figdir)
    if dn is not None:
        dn.to_csv(os.path.join(figdir, "C_denoising.csv"), index=False)
        md += ["## C. When during denoising does vision matter?\n", dn.to_markdown(index=False, floatfmt=".3f"), "\n"]

    cl, cl_tests, cl_desc = closed_loop_report(out, figdir)
    if cl is not None:
        cl.to_csv(os.path.join(figdir, "D_closed_loop.csv"), index=False)
        md += [f"## D. Closed-loop: success with patches hidden online ({cl_desc}; Wilson 95% CI)\n",
               cl.to_markdown(index=False, floatfmt=".2f"), "\n",
               "`top`/`bottom` rank patches by the signed score (so `eig:bottom` hides the most *negative* patches); "
               "`top_abs`/`bottom_abs` rank by magnitude (`bottom_abs` = least relevant patches).\n\n",
               "Paired sign tests on discordant episodes:\n",
               cl_tests.to_markdown(index=False, floatfmt=".3g"), "\n"]

    sn = sanity_report(out, figdir)
    if sn is not None:
        md += ["## E. Model-randomization sanity check (Spearman to trained-model map)\n",
               sn.rename(index=LABEL).to_markdown(floatfmt=".2f"), "\n"]

    lg, tl = language_report(out, figdir)
    if lg is not None:
        lg.to_csv(os.path.join(figdir, "F_language.csv"))
        md += ["## F. Modality reliance (Shapley share) and word attribution consistency\n",
               lg.to_markdown(floatfmt=".3f"), "\n",
               "`ig_drop_spearman`: per-joint rank agreement between |word IG| and word-removal effect. "
               "Word scores are on RDT's precomputed T5 encoder outputs, which are contextual: a token's score "
               "can carry information from the whole sentence (e.g. \"of\" in \"let go of the cube\"), so they "
               "are not a clean per-word ablation of the instruction text.\n"]
        if tl is not None:
            tl.to_csv(os.path.join(figdir, "F_modality_timeline.csv"))
            md += ["Modality share over every decision of whole successful episodes:\n",
                   tl.to_markdown(floatfmt=".3f"), "\n"]

    gr = grounding_report(out, figdir)
    if gr is not None:
        mt, po, ph, replay = gr
        mt.drop(columns="key").to_csv(os.path.join(figdir, "G_grounding_methods.csv"), index=False)
        po.to_csv(os.path.join(figdir, "G_grounding_objects.csv"), index=False)
        md += ["## G. Object grounding: what each method claims vs what removing the object does\n",
               "Each scene object (from the simulator's segmentation of the frames the policy saw; replayed "
               f"episodes match the saved frames to a mean |RGB diff| of at most {replay:.2f}) is removed "
               "from both frames and the plan change is measured (/ D_full). `claim_vs_effect_spearman`: per "
               "moment, rank agreement between a method's attribution share and the measured effect across "
               "objects (gripper, arm, task objects); `top1_agreement`: the most-claimed object is the "
               "most-effective one; `task_object_enrichment`: attribution share on task objects / their area "
               "share (median).\n\n",
               mt.drop(columns=["key", "spearman_mean"]).to_markdown(index=False, floatfmt=".3f"), "\n",
               "What the policy uses, per object (median plan change when removed vs an equally large set of "
               "background patches):\n", po.to_markdown(index=False, floatfmt=".3f"), "\n"]
        if len(ph):
            ph.to_csv(os.path.join(figdir, "G_grounding_phase.csv"), index=False)
            md += ["Before vs after the grasp (task state from the simulator): Expected IG's share and the "
                   "measured effect per task object:\n", ph.to_markdown(index=False, floatfmt=".3f"), "\n"]

    text = "\n".join(md)
    with open(os.path.join(figdir, "RESULTS.md"), "w") as f:
        f.write(text)
    return text
