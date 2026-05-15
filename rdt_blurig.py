# Feature-level BlurIG for RDT-1B
# See CITATIONS.txt for BlurIG reference (Xu et al., CVPR 2020).
#
# RDT's SigLIP encoder is frozen — no pixel→output gradient exists.
# Solution: run BlurIG in patch-embedding space (the first differentiable layer).
#
# Gradient path (all differentiable):
#   blurred SigLIP embeddings
#     → rdt.img_adaptor  (2-layer MLP with GELU)
#     → 28 RDT cross-attention blocks
#     → DPMSolver denoising loop (5 steps, prediction_type=sample)
#     → predicted action chunk (1, 64, 128)
#     → gripper-command score
#
# Score: norm of predicted gripper commands over the first 8 action steps.
# Answers: which image patches most influenced the robot's imminent grasp?
#
# ── Colab setup (run before this script) ───────────────────────────────────
# !git clone https://github.com/thu-ml/RoboticsDiffusionTransformer
# !git clone https://github.com/rdgbrandon/rdt-igtesting
# !pip install -q transformers diffusers accelerate huggingface_hub
#
# Language embeddings need T5-XXL (too large to run alongside RDT-1B on a T4).
# Pre-encode in a separate cell BEFORE loading RDT:
#
#   import sys; sys.path.insert(0, "RoboticsDiffusionTransformer")
#   from scripts.encode_lang import encode_lang_batch
#   encode_lang_batch(["pick up the object and place it in the bin"],
#                     save_path="lang_embed.pt")
#   del encode_lang_batch; import gc, torch; gc.collect(); torch.cuda.empty_cache()
#
# Then run: python rdt-igtesting/rdt_blurig.py

import os, sys
import torch
import numpy as np
import torchvision.transforms.functional as TF
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from transformers import SiglipImageProcessor, SiglipVisionModel

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE     = torch.float16 if DEVICE == "cuda" else torch.float32

RDT_REPO  = os.environ.get("RDT_REPO",    "../RoboticsDiffusionTransformer")
RDT_HF_ID = "robotics-diffusion-transformer/rdt-1b"
SIGLIP_ID = "google/siglip-so400m-patch14-384"

IMAGE_PATH = os.environ.get("ROBOT_IMAGE", "robot_frame.jpg")
LANG_PT    = os.environ.get("LANG_EMBED",  "lang_embed.pt")
TASK_TEXT  = "pick up the object and place it in the bin"

# Lower these if you run out of GPU memory.
N_DDPM_STEPS   = 5    # denoising steps per BlurIG step (DPMSolver handles 5 well)
N_BLURIG_STEPS = 20   # integration steps along the blur path
SIGMA_MAX      = 2.0  # max spatial blur sigma in patch-grid space
SCORE_HORIZON  = 8    # first N action steps to include in score

# ── RDT on sys.path ───────────────────────────────────────────────────────────
if not os.path.exists(RDT_REPO):
    raise RuntimeError(
        f"RDT repo not found at {RDT_REPO}.\n"
        "  git clone https://github.com/thu-ml/RoboticsDiffusionTransformer"
    )
sys.path.insert(0, os.path.abspath(RDT_REPO))
from models.rdt_runner import RDTRunner

print(f"Device: {DEVICE}  dtype: {DTYPE}")

# ── Language embedding ────────────────────────────────────────────────────────
# Encode with T5-XXL before loading RDT (they don't both fit on a T4 simultaneously).
# If no .pt is found, a zero-embedding fallback is used — the gradient path is still
# valid, but the attribution will be task-agnostic.
if os.path.exists(LANG_PT):
    ld = torch.load(LANG_PT, map_location="cpu")
    lang_tokens    = ld["embeddings"].to(DEVICE, dtype=DTYPE)           # (1, L, 4096)
    # attention_mask saved alongside embeddings; fall back to all-ones if absent
    lang_attn_mask = ld.get("attention_mask",
                            torch.ones(lang_tokens.shape[:2])).bool().to(DEVICE)
    print(f"Language embedding: {lang_tokens.shape}")
else:
    print(f"[warn] {LANG_PT} not found — using zero embedding (task-agnostic).")
    lang_tokens    = torch.zeros(1, 77, 4096, device=DEVICE, dtype=DTYPE)
    lang_attn_mask = torch.zeros(1, 77, dtype=torch.bool, device=DEVICE)

# ── Vision encoder ────────────────────────────────────────────────────────────
print(f"Loading {SIGLIP_ID} ...")
siglip_proc = SiglipImageProcessor.from_pretrained(SIGLIP_ID)
siglip      = SiglipVisionModel.from_pretrained(SIGLIP_ID)
siglip.to(DEVICE, dtype=DTYPE).eval()
for p in siglip.parameters():
    p.requires_grad_(False)

cfg       = siglip.config
grid_size = cfg.image_size // cfg.patch_size   # 27  (384 // 14 = 27)
n_patches = grid_size ** 2                      # 729
embed_dim = cfg.hidden_size                     # 1152
print(f"SigLIP grid: {grid_size}x{grid_size}  patches={n_patches}  dim={embed_dim}")

# ── RDT-1B ───────────────────────────────────────────────────────────────────
print(f"Loading {RDT_HF_ID} ...")
rdt = RDTRunner.from_pretrained(RDT_HF_ID)
rdt.to(DEVICE, dtype=DTYPE).eval()
# Disable grad on all weights — we differentiate w.r.t. the input embedding, not weights.
for p in rdt.parameters():
    p.requires_grad_(False)
# Override inference steps for speed (DPMSolver is accurate even at 5 steps).
rdt.num_inference_timesteps = N_DDPM_STEPS

ACTION_DIM  = rdt.action_dim    # 128 (unified state space across robot types)
PRED_HORIZ  = rdt.pred_horizon  # 64

# ── Robot state ───────────────────────────────────────────────────────────────
# Replace with real joint readings for a live robot.
# Zeros = unknown/rest pose; the model will still produce a valid prediction.
# action_mask: 1 = dimension is active for your robot platform, 0 = unused.
# For ALOHA (14-DoF): set indices [0:7] and [32:39] (or whatever the platform mapping is).
# Using all-ones here to avoid needing a platform-specific mask file.
proprio     = torch.zeros(1, 1, ACTION_DIM, device=DEVICE, dtype=DTYPE)
action_mask = torch.ones(1,  1, ACTION_DIM, device=DEVICE, dtype=DTYPE)
ctrl_freqs  = torch.tensor([25.0], device=DEVICE, dtype=DTYPE)

# ── Robot image ───────────────────────────────────────────────────────────────
if not os.path.exists(IMAGE_PATH):
    # Save a grey placeholder so the script runs end-to-end.
    # Swap for an actual robot camera frame for meaningful attributions.
    PILImage.new("RGB", (384, 384), (100, 100, 100)).save(IMAGE_PATH)
    print(f"[warn] No robot image found — saved grey placeholder to {IMAGE_PATH}.")
    print("       Replace with a real robot camera frame for meaningful results.")

frame_pil = PILImage.open(IMAGE_PATH).convert("RGB").resize((384, 384))

# ── Encode image ──────────────────────────────────────────────────────────────
def encode_image(pil_img):
    """Frozen SigLIP forward. Returns (1, n_patches, embed_dim)."""
    pv = siglip_proc(images=pil_img, return_tensors="pt")["pixel_values"]
    pv = pv.to(DEVICE, dtype=DTYPE)
    with torch.no_grad():
        return siglip(pixel_values=pv).last_hidden_state  # (1, 729, 1152)

# RDT expects 6 image views: [ext_{t-1}, rw_{t-1}, lw_{t-1}, ext_t, rw_t, lw_t]
# For a single-camera demo, broadcast the same frame across all 6 slots.
with torch.no_grad():
    single_emb = encode_image(frame_pil)         # (1, 729, 1152)
    img_tokens = single_emb.repeat(1, 6, 1)      # (1, 4374, 1152)

print(f"img_tokens shape: {img_tokens.shape}")

# ── Pre-compute frozen conditions ─────────────────────────────────────────────
# Language and state do not depend on the blurred image — compute them once.
with torch.no_grad():
    lang_cond   = rdt.lang_adaptor(lang_tokens)                       # (1, L, hidden)
    state_input = torch.cat([proprio, action_mask], dim=2)            # (1, 1, 256)
    state_traj  = rdt.state_adaptor(state_input)                      # (1, 1, hidden)

# ── Feature-level BlurIG ──────────────────────────────────────────────────────
def embed_blur(E, sigma):
    """Gaussian blur on the patch grid. E: (B, N, D) treated as (B, D, H, W)."""
    B, N, D = E.shape
    img = E.reshape(B, grid_size, grid_size, D).permute(0, 3, 1, 2)
    if sigma < 0.05:
        return img.permute(0, 2, 3, 1).reshape(B, N, D)
    ks = 2 * int(3 * sigma + 0.5) + 1
    if ks % 2 == 0:
        ks += 1
    out = TF.gaussian_blur(img.float(), kernel_size=[ks, ks], sigma=sigma).to(DTYPE)
    return out.permute(0, 2, 3, 1).reshape(B, N, D)


def rdt_score(E_t):
    """
    Run RDT's full denoising loop conditioned on a (possibly blurred) image embedding.
    Gradients flow: E_t → img_adaptor → img_cond → denoising loop → actions.

    E_t: (1, n_patches, embed_dim) with requires_grad=True
    Returns: scalar score (gripper command magnitude over first SCORE_HORIZON steps)
    """
    # Replicate across all 6 view slots — blur the whole context uniformly.
    full_img = E_t.repeat(1, 6, 1)                    # (1, 4374, 1152)
    img_cond = rdt.img_adaptor(full_img)              # (1, 4374, hidden) — grad tracks E_t

    # Run the diffusion denoising loop (N_DDPM_STEPS steps).
    # img_cond is used as cross-attention conditioning at every step, so
    # the final actions carry gradient signal all the way back to E_t.
    actions = rdt.conditional_sample(
        lang_cond, lang_attn_mask, img_cond,
        state_traj, action_mask, ctrl_freqs,
    )  # (1, PRED_HORIZ, ACTION_DIM)

    # Gripper channels: in RDT's 128-dim unified action space the gripper dims
    # are robot-specific. For ALOHA they are indices 6 and 13. Using those as
    # a proxy for "grasping intent" — the most spatially grounded action dims.
    gripper = actions[:, :SCORE_HORIZON, [6, 13]]
    return gripper.norm()


def feature_blur_ig(emb, sigma_max=SIGMA_MAX, n_steps=N_BLURIG_STEPS):
    """
    BlurIG integration along the path from maximally blurred to sharp embedding.
    Returns attribution tensor with same shape as emb: (1, n_patches, embed_dim).
    """
    sigmas = torch.linspace(sigma_max, 0.0, n_steps + 1)
    total  = torch.zeros_like(emb)

    for k in range(n_steps):
        E_t    = embed_blur(emb.detach(), sigmas[k].item()).requires_grad_(True)
        E_next = embed_blur(emb.detach(), sigmas[k + 1].item())

        score = rdt_score(E_t)
        grad  = torch.autograd.grad(score, E_t)[0]
        total = total + grad.detach() * (E_next - E_t.detach())

        if k == 0 or (k + 1) % 5 == 0:
            print(f"  step {k+1:2d}/{n_steps}  "
                  f"sigma {sigmas[k]:.2f} -> {sigmas[k+1]:.2f}  "
                  f"score={score.item():.5f}")

    return total  # (1, n_patches, embed_dim)


def to_map(attr, out_h, out_w):
    """Collapse embed_dim by L1 norm, normalize, bilinear upsample to image resolution."""
    a    = attr.squeeze(0).float().cpu().detach().numpy()   # (n_patches, embed_dim)
    grid = np.abs(a).sum(-1).reshape(grid_size, grid_size)  # (27, 27)
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    return np.array(
        PILImage.fromarray((grid * 255).astype(np.uint8))
                 .resize((out_w, out_h), PILImage.BILINEAR)
    ) / 255.0


# ── Run ───────────────────────────────────────────────────────────────────────
print(f"\nFeature-level BlurIG")
print(f"  BlurIG steps:   {N_BLURIG_STEPS}")
print(f"  DDPM steps:     {N_DDPM_STEPS}  (per BlurIG step)")
print(f"  Score:          gripper dims [6,13] over first {SCORE_HORIZON} action steps")
print(f"  Total RDT fwds: ~{N_BLURIG_STEPS * N_DDPM_STEPS}\n")

attr   = feature_blur_ig(single_emb)
img_np = np.array(frame_pil) / 255.0
amap   = to_map(attr, frame_pil.height, frame_pil.width)

# ── Visualise ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

axes[0].imshow(img_np)
axes[0].set_title("Robot camera frame", fontsize=11)

axes[1].imshow(img_np)
axes[1].imshow(amap, cmap="inferno", alpha=0.6, vmin=0, vmax=1)
axes[1].set_title("BlurIG overlay", fontsize=11)

axes[2].imshow(amap, cmap="inferno", vmin=0, vmax=1)
axes[2].set_title(f"BlurIG — gripper attribution", fontsize=11)

for ax in axes:
    ax.axis("off")

fig.suptitle(
    f"RDT-1B  —  feature-level BlurIG in SigLIP patch-embedding space\n"
    f'Task: "{TASK_TEXT}"\n'
    f"Vision: {SIGLIP_ID}  |  "
    f"Patch grid: {grid_size}x{grid_size}  |  "
    f"DDPM steps: {N_DDPM_STEPS}  |  BlurIG steps: {N_BLURIG_STEPS}",
    fontsize=9,
)
plt.tight_layout()
plt.savefig("rdt_blurig_output.png", dpi=150, bbox_inches="tight")
print("Saved: rdt_blurig_output.png")
