# Feature-level BlurIG for RDT-1B using ManiSkill simulation
# See CITATIONS.txt for BlurIG reference (Xu et al., CVPR 2020).
#
# Pipeline:
#   ManiSkill PickCube-v1 env  →  robot camera frame + joint state
#   Pre-encoded T5 lang embed  →  downloaded from HuggingFace (no T5 needed)
#   SigLIP SO400M (frozen)     →  27x27 patch embeddings
#   Feature-level BlurIG       →  gradient through img_adaptor + denoising loop
#   Output                     →  heatmap of which image patches drove gripper action
#
# Colab cells are in the repo README. Run order:
#   Cell 1: install deps
#   Cell 2: download lang embed from HuggingFace
#   Cell 3: run this script

import os, sys
os.environ.setdefault("DISPLAY", "")           # headless rendering for ManiSkill
os.environ.setdefault("MUJOCO_GL", "egl")      # force EGL backend (no X server needed)
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import torch
import numpy as np
import torchvision.transforms.functional as TF
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from transformers import SiglipImageProcessor, SiglipVisionModel

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.float16 if DEVICE == "cuda" else torch.float32

RDT_REPO  = os.environ.get("RDT_REPO", "../RoboticsDiffusionTransformer")
RDT_HF_ID = "robotics-diffusion-transformer/rdt-1b"
SIGLIP_ID = "google/siglip-so400m-patch14-384"

LANG_PT   = os.environ.get("LANG_EMBED", "lang_embed.pt")
TASK      = os.environ.get("MANISKILL_TASK", "PickCube-v1")
TASK_TEXT = "pick up the cube"

N_DDPM_STEPS   = 5
N_BLURIG_STEPS = 20
SIGMA_MAX      = 2.0
SCORE_HORIZON  = 8   # first N action steps used for gripper score

# ── RDT on sys.path ───────────────────────────────────────────────────────────
if not os.path.exists(RDT_REPO):
    raise RuntimeError(
        f"RDT repo not found at {RDT_REPO}.\n"
        "  git clone https://github.com/thu-ml/RoboticsDiffusionTransformer"
    )
sys.path.insert(0, os.path.abspath(RDT_REPO))

# huggingface_hub removed cached_download in v0.23 but older diffusers still
# import it. Patch it in before importing diffusers/RDT so the import succeeds.
import huggingface_hub as _hfhub
if not hasattr(_hfhub, "cached_download"):
    _hfhub.cached_download = _hfhub.hf_hub_download

# wandb and deepspeed are training-only. Stub them out before RDT imports so
# their broken protobuf stubs / missing CUDA extensions are never loaded.
import sys, types, importlib.machinery

def _make_stub(name):
    class _Stub(types.ModuleType):
        def __getattr__(self, n):
            # Let dunder attrs fall through to AttributeError — returning a _Stub
            # for __file__, __path__, etc. breaks os.path and importlib checks.
            if n.startswith("__") and n.endswith("__"):
                raise AttributeError(n)
            child = _make_stub(f"{self.__name__}.{n}")
            object.__setattr__(self, n, child)
            sys.modules[child.__name__] = child
            return child
        def __call__(self, *a, **kw):
            return None
        def __repr__(self):
            return f"<stub '{self.__name__}'>"

    m = _Stub(name)
    # Python 3.12 find_spec raises ValueError if __spec__ is None — use a real ModuleSpec.
    m.__spec__    = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    m.__path__    = []
    m.__file__    = ""   # empty string; os.path.splitext("") works fine
    m.__package__ = name.split(".")[0]
    return m

for _pkg in ("wandb", "deepspeed", "flash_attn"):
    for _k in [k for k in sys.modules if k == _pkg or k.startswith(_pkg + ".")]:
        del sys.modules[_k]
    sys.modules[_pkg] = _make_stub(_pkg)

from models.rdt_runner import RDTRunner
from configs.state_vec import STATE_VEC_IDX_MAPPING

# huggingface_hub >= 0.24 no longer passes `proxies`/`resume_download` to
# _from_pretrained, but RDT's CompatiblePyTorchModelHubMixin requires them.
_orig_fp = RDTRunner._from_pretrained.__func__
def _fp_compat(cls, *a, proxies=None, resume_download=False, **kw):
    return _orig_fp(cls, *a, proxies=proxies, resume_download=resume_download, **kw)
RDTRunner._from_pretrained = classmethod(_fp_compat)

print(f"Device: {DEVICE}  dtype: {DTYPE}")

# ── ManiSkill indices in the 128-dim unified state vector ─────────────────────
# Matches maniskill_model.py exactly: 7 arm joints + gripper open
MANISKILL_INDICES = (
    [STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)]
    + [STATE_VEC_IDX_MAPPING["right_gripper_open"]]
)

# Normalization stats from maniskill_model.py DATA_STAT
STATE_MIN = torch.tensor([-0.7463, -0.0801, -0.4976, -2.6578, -0.5743,  1.8310, -2.2424, 0.0000])
STATE_MAX = torch.tensor([ 0.7645,  1.4967,  0.4651, -0.3867,  0.5506,  3.2901,  2.5738, 0.0400])

# ── ManiSkill simulation ──────────────────────────────────────────────────────
print(f"Setting up ManiSkill env: {TASK} ...")
import gymnasium as gym
import mani_skill.envs   # registers the environments

env = gym.make(
    TASK,
    obs_mode="rgb",
    render_mode="rgb_array",
    sensor_configs={"width": 384, "height": 384},
)
obs, _ = env.reset(seed=42)

# Extract camera image — ManiSkill3 obs structure
sensor_data = obs.get("sensor_data", obs.get("image", {}))
cam_key     = list(sensor_data.keys())[0]             # e.g. "base_camera"
rgb         = sensor_data[cam_key]["rgb"]              # (H, W, 3) or (1, H, W, 3)
if rgb.ndim == 4:
    rgb = rgb[0]                                       # remove batch dim if present
if hasattr(rgb, "cpu"):
    rgb = rgb.cpu().numpy()
rgb = np.array(rgb, dtype=np.uint8)
frame_pil = PILImage.fromarray(rgb).resize((384, 384))
frame_pil.save("robot_frame.jpg")
print(f"Camera: {cam_key}  frame: {frame_pil.size}")

# Extract joint state — qpos is (9,) for Panda: 7 arm + 2 gripper fingers
qpos = obs["agent"]["qpos"]
if hasattr(qpos, "cpu"):
    qpos = qpos.cpu().numpy()
qpos = np.array(qpos).flatten()

# Convert 9-dim qpos to 8-dim (7 arm + gripper_open scalar)
arm_joints   = qpos[:7]
gripper_open = float(qpos[7:9].mean())                # average of two finger positions
joints_8     = np.concatenate([arm_joints, [gripper_open]])

# Normalize to [-1, 1] using ManiSkill DATA_STAT
joints_t  = torch.tensor(joints_8, dtype=torch.float32)
joints_n  = (joints_t - STATE_MIN) / (STATE_MAX - STATE_MIN).clamp(min=1e-6) * 2 - 1

# Place into 128-dim unified state vector
state_128 = torch.zeros(128, dtype=DTYPE, device=DEVICE)
state_128[MANISKILL_INDICES] = joints_n.to(DTYPE).to(DEVICE)
proprio = state_128.unsqueeze(0).unsqueeze(0)          # (1, 1, 128)

# Action mask: only the 8 ManiSkill dims are active
action_mask = torch.zeros(1, 1, 128, dtype=DTYPE, device=DEVICE)
action_mask[0, 0, MANISKILL_INDICES] = 1.0

ctrl_freqs = torch.tensor([25.0], device=DEVICE, dtype=DTYPE)
print(f"Joint state (normalized): {joints_n.numpy().round(3)}")

env.close()

# ── Language embedding ────────────────────────────────────────────────────────
# Downloaded from HuggingFace in Cell 2 — no T5 required at runtime.
if not os.path.exists(LANG_PT):
    raise FileNotFoundError(
        f"{LANG_PT} not found.\n"
        "Run Cell 2 to download the pre-encoded language embedding from HuggingFace."
    )
ld = torch.load(LANG_PT, map_location="cpu", weights_only=False)
if isinstance(ld, dict):
    lang_tokens    = ld["embeddings"]
    lang_attn_mask = ld.get("attention_mask", None)
else:
    lang_tokens    = ld          # file contains the tensor directly
    lang_attn_mask = None
if lang_tokens.dim() == 2:
    lang_tokens = lang_tokens.unsqueeze(0)   # (1, L, 4096)
lang_tokens    = lang_tokens.to(DEVICE, dtype=DTYPE)
lang_attn_mask = (lang_attn_mask.bool().to(DEVICE)
                  if lang_attn_mask is not None
                  else torch.ones(lang_tokens.shape[:2], dtype=torch.bool, device=DEVICE))
print(f"Language embedding: {lang_tokens.shape}")

# ── Vision encoder ────────────────────────────────────────────────────────────
print(f"Loading {SIGLIP_ID} ...")
siglip_proc = SiglipImageProcessor.from_pretrained(SIGLIP_ID)
siglip      = SiglipVisionModel.from_pretrained(SIGLIP_ID)
siglip.to(DEVICE, dtype=DTYPE).eval()
for p in siglip.parameters():
    p.requires_grad_(False)

cfg       = siglip.config
grid_size = cfg.image_size // cfg.patch_size   # 27
n_patches = grid_size ** 2                      # 729
embed_dim = cfg.hidden_size                     # 1152
print(f"SigLIP grid: {grid_size}x{grid_size}  patches={n_patches}  dim={embed_dim}")

# ── RDT-1B ───────────────────────────────────────────────────────────────────
print(f"Loading {RDT_HF_ID} ...")
rdt = RDTRunner.from_pretrained(RDT_HF_ID)
rdt.to(DEVICE, dtype=DTYPE).eval()
for p in rdt.parameters():
    p.requires_grad_(False)
rdt.num_inference_timesteps = N_DDPM_STEPS
if hasattr(rdt, "noise_scheduler"):
    rdt.noise_scheduler.set_timesteps(N_DDPM_STEPS)

ACTION_DIM  = rdt.action_dim    # 128
PRED_HORIZ  = rdt.pred_horizon  # 64

# ── Encode ManiSkill frame ────────────────────────────────────────────────────
def encode_image(pil_img):
    pv = siglip_proc(images=pil_img, return_tensors="pt")["pixel_values"]
    pv = pv.to(DEVICE, dtype=DTYPE)
    with torch.no_grad():
        return siglip(pixel_values=pv).last_hidden_state   # (1, 729, 1152)

# 6 view slots: [ext_{t-1}, rw_{t-1}, lw_{t-1}, ext_t, rw_t, lw_t]
# ManiSkill gives us one camera view; broadcast it across all 6 slots.
# (A real multi-camera setup would provide distinct views per slot.)
with torch.no_grad():
    single_emb = encode_image(frame_pil)         # (1, 729, 1152)
    img_tokens = single_emb.repeat(1, 6, 1)      # (1, 4374, 1152)

print(f"img_tokens: {img_tokens.shape}")

# ── Pre-compute frozen conditions ─────────────────────────────────────────────
with torch.no_grad():
    lang_cond   = rdt.lang_adaptor(lang_tokens)
    state_input = torch.cat([proprio, action_mask], dim=2)     # (1, 1, 256)
    state_traj  = rdt.state_adaptor(state_input)               # (1, 1, hidden)

# ── Feature-level BlurIG ──────────────────────────────────────────────────────
def embed_blur(E, sigma):
    B, N, D = E.shape
    img = E.reshape(B, grid_size, grid_size, D).permute(0, 3, 1, 2)
    if sigma < 0.05:
        return img.permute(0, 2, 3, 1).reshape(B, N, D)
    ks = 2 * int(3 * sigma + 0.5) + 1
    if ks % 2 == 0:
        ks += 1
    return TF.gaussian_blur(img.float(), kernel_size=[ks, ks], sigma=sigma).to(DTYPE)\
             .permute(0, 2, 3, 1).reshape(B, N, D)


def rdt_score(E_t):
    """
    Full denoising pass conditioned on blurred image embedding E_t.
    Score = norm of predicted gripper commands over the first SCORE_HORIZON steps.
    Gradient flows: E_t → img_adaptor → img_cond → denoising loop → actions.
    """
    with torch.enable_grad():
        img_cond = rdt.img_adaptor(E_t.repeat(1, 6, 1))    # (1, 4374, hidden)
        actions  = rdt.conditional_sample(
            lang_cond, lang_attn_mask, img_cond,
            state_traj, action_mask, ctrl_freqs,
        )  # (1, 64, 128)
    # MANISKILL_INDICES[6] = gripper_open (index 10 in state vec)
    gripper_idx = MANISKILL_INDICES[7]
    return actions[:, :SCORE_HORIZON, gripper_idx].norm()


def feature_blur_ig(emb):
    sigmas = torch.linspace(SIGMA_MAX, 0.0, N_BLURIG_STEPS + 1)
    total  = torch.zeros_like(emb)
    for k in range(N_BLURIG_STEPS):
        E_t    = embed_blur(emb.detach(), sigmas[k].item()).requires_grad_(True)
        E_next = embed_blur(emb.detach(), sigmas[k + 1].item())
        score  = rdt_score(E_t)
        grad   = torch.autograd.grad(score, E_t)[0]
        total  = total + grad.detach() * (E_next - E_t.detach())
        if k == 0 or (k + 1) % 5 == 0:
            print(f"  step {k+1:2d}/{N_BLURIG_STEPS}  "
                  f"sigma {sigmas[k]:.2f}->{sigmas[k+1]:.2f}  "
                  f"score={score.item():.5f}")
    return total


def to_map(attr, h, w):
    a    = attr.squeeze(0).float().cpu().detach().numpy()
    grid = np.abs(a).sum(-1).reshape(grid_size, grid_size)
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    _bilinear = getattr(PILImage, "Resampling", PILImage).BILINEAR
    return np.array(
        PILImage.fromarray((grid * 255).astype(np.uint8)).resize((w, h), _bilinear)
    ) / 255.0

# ── Run ───────────────────────────────────────────────────────────────────────
print(f"\nFeature-level BlurIG")
print(f"  Task:          {TASK}  ({TASK_TEXT})")
print(f"  BlurIG steps:  {N_BLURIG_STEPS}")
print(f"  DDPM steps:    {N_DDPM_STEPS} per BlurIG step")
print(f"  Score:         gripper (state vec idx {MANISKILL_INDICES[7]}) over {SCORE_HORIZON} steps\n")

attr   = feature_blur_ig(single_emb)
img_np = np.array(frame_pil) / 255.0
amap   = to_map(attr, frame_pil.height, frame_pil.width)

# ── Visualise ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
axes[0].imshow(img_np);                                                                    axes[0].set_title("ManiSkill frame", fontsize=11)
axes[1].imshow(img_np); axes[1].imshow(amap, cmap="inferno", alpha=0.6, vmin=0, vmax=1); axes[1].set_title("BlurIG overlay", fontsize=11)
axes[2].imshow(amap, cmap="inferno", vmin=0, vmax=1);                                     axes[2].set_title("BlurIG — gripper attribution", fontsize=11)
for ax in axes: ax.axis("off")

fig.suptitle(
    f"RDT-1B  —  feature-level BlurIG  |  Task: {TASK} ({TASK_TEXT})\n"
    f"Score: gripper command over first {SCORE_HORIZON} steps  |  "
    f"Patch grid: {grid_size}x{grid_size}  |  "
    f"BlurIG steps: {N_BLURIG_STEPS}  DDPM steps: {N_DDPM_STEPS}",
    fontsize=9,
)
plt.tight_layout()
plt.savefig("rdt_blurig_output.png", dpi=150, bbox_inches="tight")
print("\nSaved: rdt_blurig_output.png")
