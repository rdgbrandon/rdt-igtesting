"""Language and modality attribution for RDT (the repo's word x joint analysis,
rebuilt on the correct, unpadded instruction).

  words          T5 tokens of the official instruction grouped into words; checked
                 against the length of the released embedding
  word_drop      per joint, plan change when one word is removed through RDT's
                 language mask (training uses masked variable-length instructions,
                 so this is in distribution)                            W forwards / seed
  word_ig        per joint, IG over the T5 token embeddings (zero baseline, as the
                 original pipeline), all 8 joints in one backward pass   n_evals x 8 samples
  modality       exact Shapley values of vision / language / proprioception for the
                 executed plan, each removed with its trained "missing" value
                 (background image, empty-instruction embedding, centred state)  8 forwards / seed
"""
import itertools
import math

import numpy as np
import torch

from .attribution import EXEC_ROWS
from .policy import TASK2LANG

_TOKENIZER = None


def tokenizer(name="google-t5/t5-small"):
    """T5 v1.1-XXL (RDT's encoder) shares T5's 32k SentencePiece vocabulary, so
    the small checkpoint's tokenizer gives identical token ids."""
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(name)
    return _TOKENIZER


class LanguageExplainer:
    def __init__(self, policy, batch=8):
        self.p = policy
        self.batch = batch

    def _exec_all(self, A):
        return A[:, EXEC_ROWS.to(A.device)][:, :, self.p.idx]                # (B, 16, 8)

    def _noise(self, seeds):
        return torch.cat([self.p.noise(s) for s in seeds])

    # ── tokens → words ───────────────────────────────────────────────────────
    def words(self, task):
        tok = tokenizer()
        ids = tok(TASK2LANG[task]).input_ids
        L = self.p.lang(task).shape[1]
        if len(ids) != L:
            raise ValueError(f"{task}: tokenizer gives {len(ids)} tokens but the embedding has {L}")
        pieces = tok.convert_ids_to_tokens(ids)
        groups = []
        for i, pc in enumerate(pieces):
            if pc == tok.eos_token or not groups or pc.startswith("▁"):
                groups.append([i])
            else:
                groups[-1].append(i)
        labels = [tok.decode([ids[i] for i in g]).strip() or pieces[g[0]] for g in groups]
        labels = ["</s>" if pieces[g[0]] == tok.eos_token else lb for g, lb in zip(groups, labels)]
        return groups, labels

    # ── word removal ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def word_drop(self, cond, groups, seeds=(0, 1)):
        p = self.p
        L = cond["lang"].shape[1]
        masks = []
        for g in groups:
            m = torch.ones(L, dtype=torch.bool)
            m[g] = False
            masks.append(m)
        masks = torch.stack(masks)                                           # (W, L)
        full = torch.cat([self._exec_all(p.sample(cond, noise=p.noise(s))) for s in seeds])      # (S, 16, 8)
        out = torch.zeros(len(seeds), len(groups), 8)
        for si, s in enumerate(seeds):
            for i in range(0, len(groups), self.batch):
                mk = masks[i:i + self.batch]
                b = mk.shape[0]
                A = self._exec_all(p.sample(cond, emb_curr=cond["emb_curr"].expand(b, -1, -1),
                                            emb_prev=cond["emb_prev"].expand(b, -1, -1),
                                            noise=p.noise(s).expand(b, -1, -1), lang_mask=mk))
                out[si, i:i + b] = (A - full[si:si + 1]).norm(dim=1).float().cpu()   # per joint over 16 steps
        return out.mean(0).T.numpy()                                         # (8, W)

    # ── IG over token embeddings ─────────────────────────────────────────────
    def word_ig(self, cond, groups, n_evals=16, seed=0):
        p = self.p
        lang = cond["lang"]
        base = torch.zeros_like(lang)
        noise = p.noise(seed)
        with torch.no_grad():
            a1 = self._exec_all(p.sample(cond, noise=noise))
            a0 = self._exec_all(p.sample(cond, noise=noise, lang_kv=p.lang_kv(base)))
        d = (a1 - a0)[0]                                                     # (16, 8)
        u = d / d.norm(dim=0, keepdim=True).clamp_min(1e-8)                  # per-joint unit direction
        alphas = [(k + 0.5) / n_evals for k in range(n_evals)]
        jobs = [(a, j) for a in alphas for j in range(8)]
        attr = torch.zeros(8, lang.shape[1], device=p.device)
        for i in range(0, len(jobs), self.batch):
            chunk = jobs[i:i + self.batch]
            b = len(chunk)
            al = torch.tensor([a for a, _ in chunk], device=p.device, dtype=p.dtype)[:, None, None]
            x = (base + al * (lang - base)).requires_grad_(True)             # (b, L, 4096)
            sel = torch.zeros(b, 16, 8, device=p.device, dtype=p.dtype)
            for k, (_, j) in enumerate(chunk):
                sel[k, :, j] = u[:, j]
            with torch.enable_grad():
                A = self._exec_all(p.sample(cond, emb_curr=cond["emb_curr"].expand(b, -1, -1),
                                            emb_prev=cond["emb_prev"].expand(b, -1, -1),
                                            noise=noise.expand(b, -1, -1), lang_kv=p.lang_kv(x)))
                g, = torch.autograd.grad((A * sel).sum(), x)
            contrib = (g * (lang - base)).float().sum(-1) / n_evals          # (b, L)
            for k, (_, j) in enumerate(chunk):
                attr[j] += contrib[k]
        total = (a1 - a0)[0].norm(dim=0).float()                             # completeness target per joint
        tok = attr.cpu().numpy()
        words = np.stack([tok[:, g].sum(1) for g in groups], 1)              # (8, W)
        rel_err = (np.abs(tok.sum(1) - total.cpu().numpy()) / (total.cpu().numpy() + 1e-8))
        return words, rel_err

    # ── modality Shapley ─────────────────────────────────────────────────────
    @torch.no_grad()
    def modality(self, cond, seeds=(0, 1)):
        """v(S) = < A_S - A_none, u >,  u = (A_all - A_none)/||A_all - A_none||.
        Exact Shapley over {vision, language, state}; efficiency: sum = ||A_all - A_none||."""
        p = self.p
        cond_ns = p.conditions(cond["task"], cond["emb_prev"], cond["emb_curr"], cond["qpos8"],
                               lang=cond["lang"], null_state=True)
        empty_kv = p.lang_kv(p.empty_lang())
        names = ("vision", "language", "state")
        phi = {n: [] for n in names}
        for s in seeds:
            plans = {}
            for S in itertools.product([0, 1], repeat=3):
                v, l, st = S
                c = cond if st else cond_ns
                A = p.sample(c, noise=p.noise(s),
                             emb_prev=cond["emb_prev"] if v else p.bg_emb,
                             emb_curr=cond["emb_curr"] if v else p.bg_emb,
                             lang_kv=None if l else empty_kv)
                plans[S] = self._exec_all(A)[0].float()
            d = plans[(1, 1, 1)] - plans[(0, 0, 0)]
            u = d / d.norm().clamp_min(1e-8)
            val = {S: float(((plans[S] - plans[(0, 0, 0)]) * u).sum()) for S in plans}
            for i, n in enumerate(names):
                tot = 0.0
                for S in plans:
                    if S[i] == 1:
                        continue
                    k = sum(S)
                    S1 = tuple(1 if j == i else S[j] for j in range(3))
                    tot += math.factorial(k) * math.factorial(3 - k - 1) / math.factorial(3) * (val[S1] - val[S])
                phi[n].append(tot)
        return {n: float(np.mean(v)) for n, v in phi.items()}
