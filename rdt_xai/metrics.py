"""Method-agnostic faithfulness metrics for image attributions of RDT.

All perturbations happen in PIXEL space on the SigLIP input (both external
history frames, same patches), and are re-encoded by SigLIP - so no method is
favoured by sharing the metric's perturbation (as embedding-space metrics
would favour embedding-space baselines). The removal value is RDT's own
"missing camera" colour (SigLIP image_mean), i.e. normalized pixel ~0.

Effect size is the L2 deviation of the executed, normalized 8-joint plan from
the unperturbed plan, averaged over EVALUATION noise seeds that are disjoint
from the seeds used to compute the explanation.

  deletion (MoRF)   remove most-relevant patches first; AUC of D(f)/D(1)  higher = better
  deletion (LeRF)   remove least-relevant first;                          lower  = better
  insertion         start from all-removed, insert most-relevant first;
                    AUC of 1 - D(f)/D(0)                                  higher = better
"""
import numpy as np
import torch
from scipy.stats import spearmanr

FRACS = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
PATCH = 14


def _auc(y, x=FRACS):
    return float(np.trapz(y, x) / (x[-1] - x[0]))


class Faithfulness:
    def __init__(self, policy, eval_seeds=(101, 202), batch=10):
        self.p = policy
        self.eval_seeds = list(eval_seeds)
        self.batch = batch
        with torch.no_grad():
            self.bg_pix = policy.preprocess([policy.bg_image])[0]          # (3, 384, 384)

    def _patch_mask(self, order, k):
        """Boolean (384,384) pixel mask covering the first k patches in `order`."""
        G = self.p.grid
        m = torch.zeros(G * G, dtype=torch.bool)
        m[order[:k]] = True
        m = m.reshape(G, G).repeat_interleave(PATCH, 0).repeat_interleave(PATCH, 1)
        full = torch.zeros(self.bg_pix.shape[-2:], dtype=torch.bool)
        full[:m.shape[0], :m.shape[1]] = m
        if k == G * G:
            full[:] = True                                                 # incl. the 6px SigLIP ignores
        return full.to(self.p.device)

    @torch.no_grad()
    def _plans(self, cond, pix_prev, pix_curr):
        """pix_*: (N, 3, 384, 384) -> executed normalized plans (N, S, 16, 8)."""
        p = self.p
        out = []
        for i in range(0, pix_curr.shape[0], self.batch):
            pc, pp = pix_curr[i:i + self.batch], pix_prev[i:i + self.batch]
            emb = p.vision(pixel_values=torch.cat([pp, pc])).last_hidden_state
            ep, ec = emb[:pp.shape[0]], emb[pp.shape[0]:]
            per_seed = []
            for s in self.eval_seeds:
                noise = p.noise(s).expand(ec.shape[0], -1, -1)
                a = p.sample(cond, emb_curr=ec, emb_prev=ep, noise=noise)
                per_seed.append(a[:, 0:64:4][:, :, p.idx])
            out.append(torch.stack(per_seed, 1))
        return torch.cat(out).float()

    def evaluate(self, cond, img_prev, img_curr, maps):
        """maps: {method: (27,27) array}. Returns {method: metrics}."""
        p = self.p
        x_prev = p.preprocess([img_prev])[0]
        x_curr = p.preprocess([img_curr])[0]
        n = p.n_patch
        ks = [int(round(f * n)) for f in FRACS]
        ref = self._plans(cond, x_prev[None], x_curr[None])[0]                  # (S, 16, 8)
        dev = lambda P: (P - ref[None]).flatten(2).norm(dim=-1).mean(1).cpu().numpy()   # (N,)
        res = {}
        for name, m in maps.items():
            score = torch.as_tensor(np.asarray(m, dtype=np.float32).reshape(-1))
            more = torch.argsort(score, descending=True)
            less = torch.argsort(score, descending=False)
            batches = {}
            for tag, order, insert in (("morf", more, False), ("lerf", less, False), ("ins", more, True)):
                pp, pc = [], []
                for k in ks:
                    M = self._patch_mask(order, k)
                    if insert:
                        M = ~M
                    pp.append(torch.where(M, self.bg_pix, x_prev))
                    pc.append(torch.where(M, self.bg_pix, x_curr))
                batches[tag] = (torch.stack(pp), torch.stack(pc))
            pp = torch.cat([b[0] for b in batches.values()])
            pc = torch.cat([b[1] for b in batches.values()])
            d = dev(self._plans(cond, pp, pc)).reshape(3, len(ks))
            d_full = max(d[0, -1], 1e-8)                                         # everything removed
            res[name] = dict(
                deletion_morf=_auc(d[0] / d_full), deletion_lerf=_auc(d[1] / d_full),
                insertion=_auc(1 - d[2] / max(d[2, 0], 1e-8)),
                curve_morf=(d[0] / d_full).tolist(), curve_lerf=(d[1] / d_full).tolist(),
                curve_ins=(1 - d[2] / max(d[2, 0], 1e-8)).tolist(), d_full=float(d_full))
            res[name]["morf_minus_lerf"] = res[name]["deletion_morf"] - res[name]["deletion_lerf"]
        return res


def rank_agreement(a, b):
    """Spearman rank correlation of two patch maps."""
    return float(spearmanr(np.asarray(a).ravel(), np.asarray(b).ravel()).correlation)


def topk_iou(a, b, frac=0.1):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    k = max(1, int(round(frac * a.size)))
    A, B = set(np.argsort(-a)[:k]), set(np.argsort(-b)[:k])
    return len(A & B) / len(A | B)
