"""RDT-1B (ManiSkill fine-tune) policy wrapper for attribution research.

Faithful to the official evaluation path (RoboticsDiffusionTransformer/
scripts/maniskill_model.py + eval_sim/eval_rdt_maniskill.py), with three
additions that matter for attribution and for CPU budgets:

1. Explicit noise: the diffusion sampler's initial noise is an argument (or a
   seed for a private torch.Generator), never the global RNG.
2. Condition KV cache: the cross-attention keys/values of the language and
   image conditions are constant across the denoising steps, so they are
   computed once per call instead of once per step.
3. Background-slot cache: RDT receives 6 image slots
   [ext_{t-1}, rw_{t-1}, lw_{t-1}, ext_t, rw_t, lw_t]; on ManiSkill the 4 wrist
   slots are always the SigLIP-mean "missing camera" image, so their SigLIP
   embeddings and per-block K/V are computed once at load time.

The image input that attribution methods operate on is the SigLIP patch
embedding of a single external-camera slot (27x27x1152), with every other slot
held at the value the policy actually saw during the rollout.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

RDT_REPO = os.environ.get(
    "RDT_REPO",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "RoboticsDiffusionTransformer")))


if RDT_REPO not in sys.path:
    sys.path.insert(0, RDT_REPO)
import huggingface_hub as _hfhub  # noqa: E402
if not hasattr(_hfhub, "cached_download"):
    _hfhub.cached_download = _hfhub.hf_hub_download
try:                                   # RDT's Hub mixin is only for uploading/downloading checkpoints;
    import models.hub_mixin  # noqa    # newer huggingface_hub versions break its imports, so fall back
except Exception:                      # to an inert mixin (we load weights ourselves).
    import types as _types
    _hm = _types.ModuleType("models.hub_mixin")

    class CompatiblePyTorchModelHubMixin:
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__()
    _hm.CompatiblePyTorchModelHubMixin = CompatiblePyTorchModelHubMixin
    sys.modules["models.hub_mixin"] = _hm
from models.rdt_runner import RDTRunner  # noqa: E402
from configs.state_vec import STATE_VEC_IDX_MAPPING  # noqa: E402
from transformers import SiglipImageProcessor, SiglipVisionModel  # noqa: E402

# ── ManiSkill constants (scripts/maniskill_model.py) ──────────────────────────
MANISKILL_INDICES = ([STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)]
                     + [STATE_VEC_IDX_MAPPING["right_gripper_open"]])
DATA_STAT = {
    "state_min": [-0.7463043928146362, -0.0801204964518547, -0.4976441562175751, -2.657780647277832,
                  -0.5742632150650024, 1.8309762477874756, -2.2423808574676514, 0.0],
    "state_max": [0.7645499110221863, 1.4967026710510254, 0.4650936424732208, -0.3866899907588959,
                  0.5505855679512024, 3.2900545597076416, 2.5737812519073486, 0.03999999910593033],
    "action_min": [-0.7472005486488342, -0.08631071448326111, -0.4995281398296356, -2.658363103866577,
                   -0.5751323103904724, 1.8290787935256958, -2.245187997817993, -1.0],
    "action_max": [0.7654682397842407, 1.4984270334243774, 0.46786263585090637, -0.38181185722351074,
                   0.5517147779464722, 3.291581630706787, 2.575840711593628, 1.0],
}
# Exact instructions the released text_embed_*.pt files were encoded from
# (eval_sim/eval_rdt_maniskill.py). The .pt tensors are unpadded: every one of
# their L positions is a real T5 token, so the language mask is all-ones.
TASK2LANG = {
    "PegInsertionSide-v1": "Pick up a orange-white peg and insert the orange end into the box with a hole in it.",
    "PickCube-v1": "Grasp a red cube and move it to a target goal position.",
    "StackCube-v1": "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
    "PlugCharger-v1": "Pick up one of the misplaced shapes on the board/kit and insert it into the correct empty slot.",
    "PushCube-v1": "Push and move a cube to a goal region in front of it.",
}
JOINT_NAMES = ["base rot", "shoulder", "upper arm", "elbow", "forearm rot", "wrist pitch", "wrist rot", "gripper"]
CTRL_FREQ = 25
N_SLOTS = 6
EXT_PREV, EXT_CURR = 0, 3          # slot indices of the external camera at t-1 and t


def default_paths():
    data = os.environ.get("RDT_DATA", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "rdt-data")))
    return dict(
        ckpt=os.path.join(data, "rdt", "mp_rank_00_model_states.pt"),
        siglip=os.path.join(data, "siglip"),
        lang_dir=os.path.join(data, "lang"),
    )


def _expand2square(img, bg):
    w, h = img.size
    if w == h:
        return img
    s = max(w, h)
    out = Image.new(img.mode, (s, s), bg)
    out.paste(img, ((s - w) // 2, (s - h) // 2))
    return out


class RDTPolicy:
    def __init__(self, ckpt=None, siglip=None, lang_dir=None, device=None, dtype=None,
                 num_inference_steps=5, model=None, vision=None, processor=None):
        p = default_paths()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # The official eval runs bf16; attribution needs cleaner gradients, so
        # default to fp32 (TF32 tensor cores on Ampere+). RDT_DTYPE=bf16 restores it.
        if dtype is None:
            dtype = torch.bfloat16 if os.environ.get("RDT_DTYPE") == "bf16" else torch.float32
        self.dtype = dtype
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.lang_dir = lang_dir or p["lang_dir"]

        self.processor = processor or SiglipImageProcessor.from_pretrained(siglip or p["siglip"])
        self.vision = vision or SiglipVisionModel.from_pretrained(siglip or p["siglip"])
        self.vision.to(self.device, self.dtype).eval().requires_grad_(False)
        self.grid = self.vision.config.image_size // self.vision.config.patch_size     # 27
        self.n_patch = self.grid ** 2                                                   # 729
        self.img_dim = self.vision.config.hidden_size                                   # 1152

        self.rdt = model if model is not None else self._build_rdt(ckpt or p["ckpt"])
        self.rdt.to(self.device, self.dtype).eval().requires_grad_(False)
        self.rdt.num_inference_timesteps = num_inference_steps
        self.horizon = self.rdt.pred_horizon
        self.action_dim = self.rdt.action_dim

        f = lambda k: torch.tensor(DATA_STAT[k], device=self.device, dtype=self.dtype)
        self.state_min, self.state_max = f("state_min"), f("state_max")
        self.action_min, self.action_max = f("action_min"), f("action_max")
        self.idx = torch.tensor(MANISKILL_INDICES, device=self.device)
        self.elem_mask = torch.zeros(1, 1, self.action_dim, device=self.device, dtype=self.dtype)
        self.elem_mask[..., self.idx] = 1.0
        self.ctrl_freqs = torch.tensor([CTRL_FREQ], device=self.device)

        bg = tuple(int(x * 255) for x in self.processor.image_mean)
        self.bg_color = bg
        self.bg_image = Image.new("RGB", (self.processor.size["height"], self.processor.size["width"]), bg)
        with torch.no_grad():
            self.bg_emb = self.encode([self.bg_image])[0:1]                   # (1, 729, 1152)
        self._bg_kv = None
        self._lang_cache = {}

    # ── construction ─────────────────────────────────────────────────────────
    def _build_rdt(self, ckpt):
        with open(os.path.join(RDT_REPO, "configs", "base.yaml")) as fp:
            cfg = yaml.safe_load(fp)
        n_patch = 729
        img_cond_len = cfg["common"]["img_history_size"] * cfg["common"]["num_cameras"] * n_patch
        rdt = RDTRunner(
            action_dim=cfg["common"]["state_dim"], pred_horizon=cfg["common"]["action_chunk_size"],
            config=cfg["model"], lang_token_dim=cfg["model"]["lang_token_dim"],
            img_token_dim=cfg["model"]["img_token_dim"], state_token_dim=cfg["model"]["state_token_dim"],
            max_lang_cond_len=cfg["dataset"]["tokenizer_max_length"], img_cond_len=img_cond_len,
            img_pos_embed_config=[("image", (cfg["common"]["img_history_size"], cfg["common"]["num_cameras"], -n_patch))],
            lang_pos_embed_config=[("lang", -cfg["dataset"]["tokenizer_max_length"])],
            dtype=torch.float32)
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("module", sd)
        rdt.load_state_dict(sd, strict=True)
        return rdt

    # ── inputs ───────────────────────────────────────────────────────────────
    def lang(self, task):
        if task not in self._lang_cache:
            e = torch.load(os.path.join(self.lang_dir, f"text_embed_{task}.pt"), map_location="cpu", weights_only=False)
            e = e["embeddings"] if isinstance(e, dict) else e
            if e.dim() == 2:
                e = e.unsqueeze(0)
            self._lang_cache[task] = e.to(self.device, self.dtype)
        return self._lang_cache[task]

    def empty_lang(self):
        """RDT's trained "no instruction" input: the T5 encoding of "" (1 token).
        Training replaces the instruction by it with p=0.1 (train/dataset.py)."""
        if "__empty__" not in self._lang_cache:
            e = torch.load(os.path.join(RDT_REPO, "data", "empty_lang_embed.pt"), map_location="cpu", weights_only=False)
            self._lang_cache["__empty__"] = e.reshape(1, -1, e.shape[-1]).to(self.device, self.dtype)
        return self._lang_cache["__empty__"]

    def lang_kv(self, lang):
        """Per-language-block cross-attention K/V for (B, L, 4096) T5 tokens (graph-friendly)."""
        c = self.rdt.lang_adaptor(lang) + self.rdt.model.lang_cond_pos_embed[:, :lang.shape[1]]
        return [self._slot_kv(b, c) for i, b in enumerate(self.rdt.model.blocks) if i % 2 == 0]

    def preprocess(self, imgs):
        """PIL list -> SigLIP pixel tensor, exactly as maniskill_model.step does."""
        imgs = [_expand2square(self.bg_image if im is None else im.convert("RGB"), self.bg_color) for im in imgs]
        return self.processor(images=imgs, return_tensors="pt")["pixel_values"].to(self.device, self.dtype)

    @torch.no_grad()
    def encode(self, imgs):
        return self.vision(pixel_values=self.preprocess(imgs)).last_hidden_state   # (N, 729, 1152)

    def state_token(self, qpos8):
        """qpos8: raw [7 arm joints, first finger] (= obs['agent']['qpos'][:, :-1])."""
        j = torch.as_tensor(np.asarray(qpos8, dtype=np.float32).reshape(-1)[:8], device=self.device, dtype=self.dtype)
        jn = (j - self.state_min) / (self.state_max - self.state_min) * 2 - 1
        s = torch.zeros(1, 1, self.action_dim, device=self.device, dtype=self.dtype)
        s[0, 0, self.idx] = jn
        return s

    def unnormalize(self, a):
        """(…, 128) normalized RDT action -> (…, 8) joint targets in env units."""
        a = a[..., self.idx]
        return (a + 1) / 2 * (self.action_max - self.action_min) + self.action_min

    # ── cached conditional sampling ──────────────────────────────────────────
    def _slot_kv(self, block, c_slot):
        """K/V of one block's cross-attention for a chunk of condition tokens
        (already adapted and position-embedded). RmsNorm on K is per token, so
        K/V of a concatenated condition = concatenation of per-chunk K/V."""
        ca = block.cross_attn
        B, L, _ = c_slot.shape
        kv = ca.kv(c_slot).reshape(B, L, 2, ca.num_heads, ca.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        return ca.k_norm(k), v

    def _img_blocks(self):
        return [b for i, b in enumerate(self.rdt.model.blocks) if i % 2 == 1]

    def _bg_slot_kv(self):
        """Per-image-block K/V for the 4 wrist slots, which always hold bg_emb."""
        if self._bg_kv is None:
            with torch.no_grad():
                adapted = self.rdt.img_adaptor(self.bg_emb)                        # (1, 729, D)
                pos = self.rdt.model.img_cond_pos_embed
                kvs = {}
                for slot in range(N_SLOTS):
                    if slot in (EXT_PREV, EXT_CURR):
                        continue
                    c = adapted + pos[:, slot * self.n_patch:(slot + 1) * self.n_patch]
                    kvs[slot] = [self._slot_kv(b, c) for b in self._img_blocks()]
                self._bg_kv = kvs
        return self._bg_kv

    def conditions(self, task, emb_prev, emb_curr, qpos8, lang=None, null_state=False):
        """Everything except the external-camera embeddings that attribution
        may perturb: returns a dict reused by `sample`.
        lang: override the instruction tokens (e.g. empty_lang()).
        null_state: proprioception at the centre of the normalized range (0),
        approximating training's mean-state dropout."""
        lang = self.lang(task) if lang is None else lang
        with torch.no_grad():
            lang_kv = self.lang_kv(lang)
            st = self.state_token(qpos8)
            if null_state:
                st = torch.zeros_like(st)
            state_traj = self.rdt.state_adaptor(torch.cat([st, self.elem_mask], dim=2))
        return dict(lang_kv=lang_kv, lang=lang, state_traj=state_traj, qpos8=np.asarray(qpos8),
                    emb_prev=emb_prev, emb_curr=emb_curr, task=task)

    def _ext_kv(self, emb, slot):
        c = self.rdt.img_adaptor(emb) + self.rdt.model.img_cond_pos_embed[:, slot * self.n_patch:(slot + 1) * self.n_patch]
        return [self._slot_kv(b, c) for b in self._img_blocks()]

    def noise(self, seed, batch=1):
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        return torch.randn(batch, self.horizon, self.action_dim, generator=g).to(self.device, self.dtype)

    def _bg_ext_kv(self):
        """K/V of the background embedding placed in the two external slots:
        used to knock vision out of selected denoising steps."""
        if getattr(self, "_bg_ext", None) is None:
            with torch.no_grad():
                self._bg_ext = (self._ext_kv(self.bg_emb, EXT_PREV), self._ext_kv(self.bg_emb, EXT_CURR))
        return self._bg_ext

    def _img_kv(self, prev_kv, curr_kv, B):
        bg = self._bg_slot_kv()
        img_kv = []
        for bi in range(len(curr_kv)):
            parts = []
            for slot in range(N_SLOTS):
                k, v = prev_kv[bi] if slot == EXT_PREV else curr_kv[bi] if slot == EXT_CURR else bg[slot][bi]
                parts.append((k.expand(B, -1, -1, -1), v.expand(B, -1, -1, -1)))
            img_kv.append((torch.cat([p[0] for p in parts], 2), torch.cat([p[1] for p in parts], 2)))
        return img_kv

    def sample(self, cond, emb_curr=None, emb_prev=None, noise=None, seed=0, return_attn=False,
               attn_grad=False, vision_steps=None, step_embs=None, lang_kv=None, lang_mask=None):
        """Denoise an action chunk. Returns normalized actions (B, 64, 128).

        emb_curr/emb_prev: override the external-camera embeddings (B, 729, 1152); may require grad.
        return_attn:  also return per (step, image block) cross-attention, head-averaged (B, N, L),
                      or with attn_grad=True the raw (B, H, N, L) weights with retain_grad() set.
        vision_steps: denoising-step indices that see the real external view; the rest see the
                      background ("camera missing") embedding. None = all steps.
        step_embs:    list over steps of (emb_prev_t, emb_curr_t): a separate leaf per step so one
                      backward pass splits the input gradient by denoising step.
        lang_kv:      override the language K/V (from lang_kv(), may carry grad; batch 1 or B).
        lang_mask:    (B, L) bool, False = token removed (RDT is trained with masked, variable-length
                      instructions, so this is an in-distribution deletion).
        """
        emb_curr = cond["emb_curr"] if emb_curr is None else emb_curr
        emb_prev = cond["emb_prev"] if emb_prev is None else emb_prev
        B = step_embs[0][1].shape[0] if step_embs is not None else emb_curr.shape[0]
        if noise is None:
            noise = self.noise(seed, B)
        n_steps = self.rdt.num_inference_timesteps
        if step_embs is not None:
            img_kv_steps = [self._img_kv(self._ext_kv(p, EXT_PREV), self._ext_kv(c, EXT_CURR), B) for p, c in step_embs]
        else:
            real = self._img_kv(self._ext_kv(emb_prev, EXT_PREV), self._ext_kv(emb_curr, EXT_CURR), B)
            img_kv_steps = [real] * n_steps
            if vision_steps is not None:
                off = self._img_kv(*self._bg_ext_kv(), B)
                img_kv_steps = [real if s in set(vision_steps) else off for s in range(n_steps)]
        lang_kv = [(k.expand(B, -1, -1, -1), v.expand(B, -1, -1, -1)) for k, v in (lang_kv or cond["lang_kv"])]
        lmask = None if lang_mask is None else lang_mask.to(self.device).reshape(B, 1, 1, -1)
        state_traj = cond["state_traj"].expand(B, -1, -1)

        rdt, model = self.rdt, self.rdt.model
        sched = rdt.noise_scheduler_sample
        sched.set_timesteps(n_steps)
        mask = self.elem_mask.expand(B, self.horizon, -1)
        x_act = noise
        freq = model.freq_embedder(self.ctrl_freqs.expand(B)).unsqueeze(1)
        attn_log = [] if return_attn else None
        for si, t in enumerate(sched.timesteps):
            img_kv = img_kv_steps[si]
            a = rdt.state_adaptor(torch.cat([x_act, mask], dim=2))
            x = torch.cat([state_traj, a], dim=1)
            te = model.t_embedder(t.unsqueeze(-1).to(self.device)).unsqueeze(1).expand(B, -1, -1)
            h = torch.cat([te, freq, x], dim=1) + model.x_pos_embed
            for i, blk in enumerate(model.blocks):
                k, v = lang_kv[i // 2] if i % 2 == 0 else img_kv[i // 2]
                if i % 2 == 0:
                    h = self._block(blk, h, k, v, mask=lmask)
                else:
                    h = self._block(blk, h, k, v, attn_log, attn_grad)
            out = model.final_layer(h)[:, -self.horizon:]
            x_act = sched.step(out, t, x_act).prev_sample.to(self.dtype)
        x_act = x_act * mask
        return (x_act, attn_log) if return_attn else x_act

    @staticmethod
    def _block(blk, x, k, v, attn_log=None, attn_grad=False, mask=None):
        x = x + blk.attn(blk.norm1(x))
        ca = blk.cross_attn
        y = blk.norm2(x)
        B, N, C = y.shape
        q = ca.q_norm(ca.q(y).reshape(B, N, ca.num_heads, ca.head_dim).permute(0, 2, 1, 3))
        if attn_log is not None:
            w = torch.softmax((q * ca.scale) @ k.transpose(-2, -1), dim=-1)
            if attn_grad and w.requires_grad:
                w.retain_grad()
                attn_log.append(w)                                              # (B, H, N, L)
            else:
                attn_log.append(w.mean(1).detach())                             # (B, N, L)
            o = w @ v
        else:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + ca.proj(o.permute(0, 2, 1, 3).reshape(B, N, C))
        return x + blk.ffn(blk.norm3(x))

    # ── convenience: one policy step as in eval_rdt_maniskill.py ─────────────
    @torch.no_grad()
    def act(self, task, img_prev, img_curr, qpos8, seed=0, stride=4):
        embs = self.encode([img_prev, img_curr])
        cond = self.conditions(task, embs[0:1], embs[1:2], qpos8)
        a = self.sample(cond, seed=seed)
        joints = self.unnormalize(a)[0].float().cpu().numpy()                   # (64, 8)
        return joints[::stride], cond, a
