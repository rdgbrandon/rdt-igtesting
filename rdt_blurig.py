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
DTYPE    = torch.bfloat16 if DEVICE == "cuda" else torch.float32

RDT_REPO  = os.environ.get("RDT_REPO", "../RoboticsDiffusionTransformer")
RDT_HF_ID = "robotics-diffusion-transformer/rdt-1b"
SIGLIP_ID = "google/siglip-so400m-patch14-384"

LANG_PT   = os.environ.get("LANG_EMBED", "lang_embed.pt")
TASK      = os.environ.get("MANISKILL_TASK", "PickCube-v1")
_TASK_LABELS = {
    "PickCube-v1":         "pick up the cube",
    "StackCube-v1":        "stack the cubes",
    "PushCube-v1":         "push the cube",
    "PegInsertionSide-v1": "insert the peg",
    "PlugCharger-v1":      "plug the charger",
}
_TASK_DESCRIPTIONS = {
    "PickCube-v1":         "Grasp a red cube and move it to a target goal position.",
    "StackCube-v1":        "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
    "PushCube-v1":         "Push and move a cube to a goal region in front of it.",
    "PegInsertionSide-v1": "Pick up a orange-white peg and insert the orange end into the box with a hole in it.",
    "PlugCharger-v1":      "Pick up one of the misplaced shapes on the board/kit and insert it into the correct empty slot.",
}
TASK_TEXT        = _TASK_LABELS.get(TASK, TASK)
TASK_DESCRIPTION = _TASK_DESCRIPTIONS.get(TASK, TASK_TEXT)

N_DDPM_STEPS   = 5
N_BLURIG_STEPS = 20
SIGMA_MAX      = 2.0
SCORE_HORIZON  = 8   # first N action steps used for gripper score
SKIP_IG        = os.environ.get("SKIP_IG", "0") == "1"

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

# Clear RDT module cache so %run always gets a fresh, unpatched class.
# Without this, the second %run captures the already-patched _from_pretrained
# as _orig_fp and recurses infinitely.
for _k in list(sys.modules):
    if _k == "models" or _k.startswith("models.") or \
       _k == "configs" or _k.startswith("configs."):
        del sys.modules[_k]

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
STATE_MIN  = torch.tensor([-0.7463, -0.0801, -0.4976, -2.6578, -0.5743,  1.8310, -2.2424,  0.0000])
STATE_MAX  = torch.tensor([ 0.7645,  1.4967,  0.4651, -0.3867,  0.5506,  3.2901,  2.5738,  0.0400])
# Action range differs from state range — gripper is [-1, 1] in action space, not [0, 0.04]
ACTION_MIN = torch.tensor([-0.7472, -0.0863, -0.4995, -2.6584, -0.5751,  1.8291, -2.2452, -1.0000])
ACTION_MAX = torch.tensor([ 0.7655,  1.4984,  0.4679, -0.3818,  0.5517,  3.2916,  2.5758,  1.0000])

# ── ManiSkill simulation ──────────────────────────────────────────────────────
print(f"Setting up ManiSkill env: {TASK} ...")
import gymnasium as gym
import mani_skill.envs   # registers the environments

# ── Wrist camera environment factory ─────────────────────────────────────────
_WRIST_REGISTERED = set()

# Direct module paths for each task — avoids gym registry entry_point parsing
_TASK_MODULES = {
    "PickCube-v1":         ("mani_skill.envs.tasks.tabletop.pick_cube",          "PickCubeEnv"),
    "StackCube-v1":        ("mani_skill.envs.tasks.tabletop.stack_cube",         "StackCubeEnv"),
    "PushCube-v1":         ("mani_skill.envs.tasks.tabletop.push_cube",          "PushCubeEnv"),
    "PegInsertionSide-v1": ("mani_skill.envs.tasks.tabletop.peg_insertion_side", "PegInsertionSideEnv"),
    "PlugCharger-v1":      ("mani_skill.envs.tasks.tabletop.plug_charger",       "PlugChargerEnv"),
}

def make_env_with_wrist(task_id, **kwargs):
    """
    Create a ManiSkill3 env with a wrist camera mounted on panda_hand_tcp.
    Falls back to the base env (base_camera only) if camera mounting fails.
    """
    import importlib, traceback as _tb
    wrist_id = task_id.replace("-v", "Wrist-v")

    if wrist_id not in _WRIST_REGISTERED:
        # Purge any stale registration from a previous %run so we always re-register
        if wrist_id in gym.envs.registry:
            del gym.envs.registry[wrist_id]
        try:
            from mani_skill.sensors.camera import CameraConfig
            import sapien as _sapien

            # Resolve base class via direct import first, then gym registry
            base_cls = None
            if task_id in _TASK_MODULES:
                mod_path, cls_name = _TASK_MODULES[task_id]
                try:
                    base_cls = getattr(importlib.import_module(mod_path), cls_name)
                except Exception:
                    pass
            if base_cls is None:
                spec = gym.envs.registry.get(task_id)
                if spec is None:
                    raise ValueError(f"{task_id!r} not in gym registry")
                ep = spec.entry_point
                if isinstance(ep, str):
                    mod_path, cls_name = ep.rsplit(":", 1)
                    base_cls = getattr(importlib.import_module(mod_path), cls_name)
                else:
                    base_cls = ep

            # Mount wrist camera via _setup_sensors override using mount=link.
            # camera.py has a bug: when entity_uid is used but articulation=None,
            # self.entity is never set (line 144-145 does `pass` instead of
            # `self.entity = None`), causing AttributeError at line 152.
            # Using mount= hits line 139-140 which works correctly.
            class _WristEnv(base_cls):
                def _setup_sensors(self, *args, **kwargs):
                    super()._setup_sensors(*args, **kwargs)
                    try:
                        from mani_skill.sensors.camera import CameraConfig as _CC
                        from mani_skill.sensors.camera import Camera as _Cam
                        import numpy as _np2

                        # Find TCP/hand link
                        _hand = None
                        for _lnk in self.agent.robot.get_links():
                            if _lnk.name in ('panda_hand_tcp', 'tcp', 'panda_hand'):
                                _hand = _lnk; break
                        if _hand is None:
                            for _lnk in self.agent.robot.get_links():
                                if 'tcp' in _lnk.name.lower() or 'hand' in _lnk.name.lower():
                                    _hand = _lnk; break
                        if _hand is None:
                            _names = [l.name for l in self.agent.robot.get_links()]
                            print(f"Wrist cam: no TCP link. Links: {_names}")
                            return

                        # Identity local pose — camera sits at the TCP frame origin
                        _lp = None
                        for _pf in [
                            lambda: _sapien.Pose(p=[0,0,0], q=[1,0,0,0]),
                            lambda: _sapien.Pose([0,0,0], [1,0,0,0]),
                            lambda: _sapien.Pose(_np2.array([0,0,0,1,0,0,0], dtype='float32')),
                            lambda: _sapien.Pose(_np2.zeros(7, dtype='float32')),
                            lambda: _sapien.Pose(),
                        ]:
                            try: _lp = _pf(); break
                            except: continue
                        if _lp is None:
                            print("Wrist cam: cannot construct Pose"); return

                        _cfg = _CC(uid="wrist_camera", pose=_lp,
                                   width=384, height=384, fov=1.57,
                                   near=0.01, far=100, mount=_hand)
                        _cam = _Cam(_cfg, self.scene)
                        self._sensors['wrist_camera'] = _cam
                        print(f"Wrist camera mounted on: {_hand.name}")
                    except Exception:
                        import traceback as _tb2; _tb2.print_exc()

            _WristEnv.__name__     = f"{base_cls.__name__}Wrist"
            _WristEnv.__qualname__ = _WristEnv.__name__

            spec = gym.envs.registry.get(task_id)
            ms   = getattr(spec, 'max_episode_steps', None) or 200
            gym.register(wrist_id, entry_point=_WristEnv, max_episode_steps=ms)
            _WRIST_REGISTERED.add(wrist_id)
            print(f"Registered wrist-camera env: {wrist_id}")

        except Exception as _e:
            print(f"Wrist camera registration failed — falling back to {task_id}")
            _tb.print_exc()
            _WRIST_REGISTERED.add(wrist_id)
            return gym.make(task_id, **kwargs)

    use_id = wrist_id if wrist_id in gym.envs.registry else task_id
    return gym.make(use_id, **kwargs)


env = make_env_with_wrist(
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
if lang_attn_mask is not None:
    lang_attn_mask = lang_attn_mask.bool().to(DEVICE)
else:
    # .pt file has no mask — infer from tokenizing the task description
    from transformers import AutoTokenizer as _AutoTok
    _t5tok = _AutoTok.from_pretrained("t5-small")
    _n_real = min(len(_t5tok(TASK_DESCRIPTION).input_ids), lang_tokens.shape[1])
    lang_attn_mask = torch.zeros(lang_tokens.shape[:2], dtype=torch.bool, device=DEVICE)
    lang_attn_mask[0, :_n_real] = True
    print(f"Language mask inferred: {_n_real} real tokens / {lang_tokens.shape[1]} total")
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

# Try to overlay ManiSkill fine-tuned weights on top of the base model.
# The maniskill-model repo contains a DeepSpeed checkpoint (mp_rank_00_model_states.pt)
# that was fine-tuned specifically on these 5 ManiSkill tasks.
try:
    from huggingface_hub import hf_hub_download as _hf_dl
    _ckpt_path = _hf_dl("robotics-diffusion-transformer/maniskill-model",
                         "rdt/mp_rank_00_model_states.pt")
    _ckpt = torch.load(_ckpt_path, map_location="cpu", weights_only=False)
    _sd   = _ckpt.get("module", _ckpt)   # DeepSpeed wraps weights in 'module'
    missing, unexpected = rdt.load_state_dict(_sd, strict=False)
    print(f"ManiSkill fine-tuned weights loaded "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
except Exception as _e:
    print(f"ManiSkill fine-tune not loaded ({_e}) — using base rdt-1b weights")

rdt.to(DEVICE, dtype=DTYPE).eval()
for p in rdt.parameters():
    p.requires_grad_(False)
rdt.num_inference_timesteps = N_DDPM_STEPS
# RDT uses DPMSolverMultistepScheduler for inference, DDPMScheduler for training
for _sched_attr in ("noise_scheduler_sample", "noise_scheduler"):
    if hasattr(rdt, _sched_attr):
        getattr(rdt, _sched_attr).set_timesteps(N_DDPM_STEPS)

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


IG_SEED = 0   # fixed seed for the diffusion sampler's initial noise (see rdt_score)

def rdt_score(E_t):
    """
    Full denoising pass conditioned on blurred image embedding E_t.
    Score = norm of predicted gripper commands over the first SCORE_HORIZON steps.
    Gradient flows: E_t → img_adaptor → img_cond → denoising loop → actions.

    rdt.conditional_sample starts from random Gaussian noise and denoises it —
    without a fixed seed, every call (one per BlurIG step) draws a *different*
    noise sample, so score differences between steps would be confounded by
    random noise, not just the blur change we're trying to measure. Re-seeding
    right before the call ensures only E_t differs between calls.
    """
    torch.manual_seed(IG_SEED)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(IG_SEED)
    with torch.enable_grad():
        img_cond = rdt.img_adaptor(E_t.repeat(1, 6, 1))    # (1, 4374, hidden)
        actions  = rdt.conditional_sample(
            lang_cond, lang_attn_mask, img_cond,
            state_traj, action_mask, ctrl_freqs,
        )  # (1, 64, 128)
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
if SKIP_IG:
    print("Models loaded. SKIP_IG=1 — skipping BlurIG. Ready.")
else:
    print(f"\nFeature-level BlurIG")
    print(f"  Task:          {TASK}  ({TASK_TEXT})")
    print(f"  BlurIG steps:  {N_BLURIG_STEPS}")
    print(f"  DDPM steps:    {N_DDPM_STEPS} per BlurIG step")
    print(f"  Score:         gripper over {SCORE_HORIZON} steps\n")

    attr   = feature_blur_ig(single_emb)
    img_np = np.array(frame_pil) / 255.0

    # Raw 27×27 patch grid (before bilinear upsampling)
    a_np     = attr.squeeze(0).float().cpu().detach().numpy()
    raw_grid = np.abs(a_np).sum(-1).reshape(grid_size, grid_size)
    raw_grid = (raw_grid - raw_grid.min()) / (raw_grid.max() - raw_grid.min() + 1e-8)
    amap     = to_map(attr, frame_pil.height, frame_pil.width)

    import matplotlib.cm as _mcm
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))

    axes[0].imshow(img_np)
    axes[0].set_title("ManiSkill frame", fontsize=11)

    # Attribution-proportional alpha: low-signal regions stay transparent
    axes[1].imshow(img_np)
    _rgba = _mcm.get_cmap("inferno")(amap)
    _rgba[..., 3] = np.sqrt(amap) * 0.85
    axes[1].imshow(_rgba)
    axes[1].set_title("BlurIG overlay", fontsize=11)

    axes[2].imshow(amap, cmap="inferno", vmin=0, vmax=1)
    axes[2].set_title("BlurIG — gripper attribution", fontsize=11)

    # Raw patch grid at native 27×27 resolution
    im4 = axes[3].imshow(raw_grid, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
    axes[3].set_title(f"Patch grid  ({grid_size}×{grid_size})", fontsize=11)
    fig.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)

    for ax in axes[:3]:
        ax.axis("off")
    fig.suptitle(
        f"RDT-1B  —  feature-level BlurIG  |  Task: {TASK} ({TASK_TEXT})\n"
        f"Score: gripper over first {SCORE_HORIZON} steps  |  "
        f"Patch grid: {grid_size}×{grid_size}  |  "
        f"BlurIG steps: {N_BLURIG_STEPS}  DDPM steps: {N_DDPM_STEPS}",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig("rdt_blurig_output.png", dpi=150, bbox_inches="tight")
    print("\nSaved: rdt_blurig_output.png")
