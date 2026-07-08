#!/usr/bin/env python3
"""
Local (Windows) rollout runner — fully independent of the Colab files.

Mirrors rollout_subprocess.py's rollout logic but with local paths instead of
/content/... and without the Colab pip self-heal (the local venv is managed).
Does NOT modify or import rollout_subprocess.py / run_rollout_cell.py.

Usage (from the IG_RDT directory, inside the local venv):
    python local_rollout.py --task PickCube-v1 --n 1
"""
import os, sys, argparse

BASE = os.path.dirname(os.path.abspath(__file__))
RDT_REPO_DEFAULT = os.path.abspath(os.path.join(BASE, '..', 'RoboticsDiffusionTransformer'))

parser = argparse.ArgumentParser()
parser.add_argument('--task',         default='PickCube-v1')
parser.add_argument('--n',            type=int, default=1,   help='number of successes to collect')
parser.add_argument('--max-attempts', type=int, default=500, help='episode cap to avoid infinite loop')
parser.add_argument('--base-seed',    type=int, default=20241201)
parser.add_argument('--rdt-repo',     default=RDT_REPO_DEFAULT)
parser.add_argument('--frames', type=int, default=30, help='evenly spaced frames to save per success')
parser.add_argument('--show',   action='store_true', help='display each frame live in an OpenCV window')
parser.add_argument('--video',  action='store_true', help='save an .mp4 of every episode')
parser.add_argument('--fps',    type=int, default=20, help='fps for --show pacing and --video output')
args = parser.parse_args()

os.environ.setdefault('DISPLAY', '')
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ['RDT_REPO']       = args.rdt_repo
os.environ['SKIP_IG']        = '1'
os.environ['MANISKILL_TASK'] = args.task
os.environ['LANG_EMBED']     = f'lang_embed_{args.task}.pt'

sys.path.insert(0, args.rdt_repo)
os.chdir(BASE)

if not os.path.exists(os.environ['LANG_EMBED']):
    os.environ['LANG_EMBED'] = 'lang_embed.pt'

print(f'LOCAL: task={args.task}  n={args.n}  base_seed={args.base_seed}', flush=True)

# ManiSkill's default GPU-direct render transfer (sapien_cuda) crashes with an
# access violation on Windows/WDDM. Wrap gym.make so every env created here or
# inside rdt_blurig.py uses the CPU-copy render path instead — this keeps
# rdt_blurig.py itself unmodified.
import gymnasium as _gymn
_orig_gym_make = _gymn.make
def _gym_make_cpu(*a, **kw):
    kw.setdefault('render_backend', 'cpu')
    return _orig_gym_make(*a, **kw)
_gymn.make = _gym_make_cpu

print('LOCAL: Loading models (from HF cache) ...', flush=True)

import runpy
_g = runpy.run_path(os.path.join(BASE, 'rdt_blurig.py'), run_name='__main__')

print('LOCAL: Models loaded.', flush=True)

import torch, numpy as np
from PIL import Image as PILImage
from collections import deque
import gymnasium as gym
import mani_skill.envs

rdt               = _g['rdt']
siglip            = _g['siglip']
siglip_proc       = _g['siglip_proc']
lang_tokens       = _g['lang_tokens']
lang_attn_mask    = _g['lang_attn_mask']
MANISKILL_INDICES = _g['MANISKILL_INDICES']
STATE_MIN         = _g['STATE_MIN']
STATE_MAX         = _g['STATE_MAX']
ACTION_MIN        = _g['ACTION_MIN']
ACTION_MAX        = _g['ACTION_MAX']
DEVICE            = _g['DEVICE']
DTYPE             = _g['DTYPE']
N_DDPM_STEPS      = _g['N_DDPM_STEPS']
embed_dim         = _g['embed_dim']

_bg_color = tuple(int(x * 255) for x in siglip_proc.image_mean)
_bg_img   = PILImage.new('RGB', (384, 384), _bg_color)
_use_pa   = hasattr(rdt, 'predict_action')
print(f'LOCAL: bg={_bg_color}  predict_action={_use_pa}  DDPM_steps={N_DDPM_STEPS}', flush=True)

STATE_MIN_t  = STATE_MIN.to(DEVICE, dtype=DTYPE)
STATE_MAX_t  = STATE_MAX.to(DEVICE, dtype=DTYPE)
ACTION_MIN_t = ACTION_MIN.to(DEVICE, dtype=DTYPE)
ACTION_MAX_t = ACTION_MAX.to(DEVICE, dtype=DTYPE)


def _expand2square(img, bg):
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    out = PILImage.new(img.mode, (side, side), bg)
    out.paste(img, ((side - w) // 2, (side - h) // 2))
    return out

def _encode_6(pil_list):
    bg = tuple(int(x * 255) for x in siglip_proc.image_mean)
    imgs = [_expand2square(_bg_img if img is None else img, bg) for img in pil_list]
    pvs  = siglip_proc(images=imgs, return_tensors='pt')['pixel_values'].to(DEVICE, dtype=DTYPE)
    with torch.no_grad():
        embs = siglip(pixel_values=pvs).last_hidden_state
    return embs.reshape(1, -1, embed_dim)


if args.show:
    import cv2

def _show_frame(pil_img, ep_idx, step):
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv2.putText(bgr, f'ep {ep_idx}  step {step}', (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow(f'{args.task}', bgr)
    cv2.waitKey(max(1, 1000 // args.fps))


def _save_video(frames, ep_idx, success):
    import imageio
    path = f'rollout_ep{ep_idx:02d}_seed{ep_idx+args.base_seed}_{"success" if success else "fail"}.mp4'
    imageio.mimsave(path, [np.array(f) for f in frames], fps=args.fps, macro_block_size=1)
    print(f'LOCAL: wrote {path} ({len(frames)} frames)', flush=True)


def _render_pil(env):
    r = env.render()
    if hasattr(r, 'cpu'): r = r.cpu().numpy()
    img = PILImage.fromarray(np.array(r).squeeze().astype(np.uint8))
    if img.width != img.height:
        print(f'LOCAL: non-square render {img.width}x{img.height}', flush=True)
    return img


def _make_env(max_ep=400):
    _e = gym.make(args.task, obs_mode='rgb', render_mode='rgb_array',
                  control_mode='pd_joint_pos', render_backend='cpu')
    _w = _e
    while _w is not None:
        if hasattr(_w, '_max_episode_steps'):
            _w._max_episode_steps = max_ep; break
        _w = getattr(_w, 'env', None)
    return _e


def _get_joint8(obs):
    """Raw (unnormalized) 8-dim joint state [7 arm + gripper] as a plain list,
    so it can be saved to JSON and later renormalized for IG analysis."""
    _qp = obs['agent']['qpos']
    if hasattr(_qp, 'cpu'): _qp = _qp.cpu()
    return np.array(_qp).flatten()[:8].tolist()


def rollout(ep_idx):
    _env = _make_env()
    _obs, _ = _env.reset(seed=ep_idx + args.base_seed)
    first_frame = _render_pil(_env)
    _hist  = deque([None, first_frame], maxlen=2)
    if args.show:
        _show_frame(first_frame, ep_idx, 0)
    frames    = [first_frame]
    qpos_hist = [_get_joint8(_obs)]
    chunk, cp, done, step, info = None, 16, False, 0, {}

    while not done and step < 400:
        if cp >= 16:
            _raw = _encode_6([_hist[0], None, None, _hist[1], None, None])
            _qp  = _obs['agent']['qpos']
            if hasattr(_qp, 'cpu'): _qp = _qp.cpu()
            _j8  = torch.tensor(np.array(_qp).flatten()[:8], dtype=DTYPE, device=DEVICE).unsqueeze(0)
            _jn  = (_j8 - STATE_MIN_t) / (STATE_MAX_t - STATE_MIN_t).clamp(min=1e-6) * 2 - 1
            _st  = torch.zeros(1, 1, 128, dtype=DTYPE, device=DEVICE)
            _st[0, 0, MANISKILL_INDICES] = _jn[0]
            _mk  = torch.zeros(1, 128, dtype=DTYPE, device=DEVICE)
            _mk[0, MANISKILL_INDICES] = 1.0
            _cf  = torch.tensor([25.0], device=DEVICE, dtype=DTYPE)
            with torch.no_grad():
                if _use_pa:
                    _tr = rdt.predict_action(
                        lang_tokens=lang_tokens, lang_attn_mask=lang_attn_mask,
                        img_tokens=_raw, state_tokens=_st,
                        action_mask=_mk.unsqueeze(1), ctrl_freqs=_cf)
                else:
                    _ic = rdt.img_adaptor(_raw)
                    _sc = rdt.state_adaptor(torch.cat([_st, _mk.unsqueeze(1)], dim=2))
                    _tr = rdt.conditional_sample(
                        _g['lang_cond'], lang_attn_mask, _ic, _sc, _mk.unsqueeze(1), _cf)
            _acts = (_tr[0, :, MANISKILL_INDICES] + 1) / 2 * (ACTION_MAX_t - ACTION_MIN_t) + ACTION_MIN_t
            chunk = _acts[::4].cpu().float().numpy()
            cp = 0

        _obs, _, term, trunc, info = _env.step(chunk[cp].reshape(1, 8))
        f = _render_pil(_env)
        if args.show:
            _show_frame(f, ep_idx, step + 1)
        _hist.append(f)
        frames.append(f)
        qpos_hist.append(_get_joint8(_obs))
        done = bool(term) or bool(trunc)
        cp += 1; step += 1

    _env.close()
    success = bool(info.get('success', False))
    if args.video:
        _save_video(frames, ep_idx, success)
    if success:
        n_total = len(frames)
        indices = [int(round(i)) for i in np.linspace(0, n_total - 1, args.frames)]
        frame_paths = []
        for k, idx in enumerate(indices):
            path = f'success_ep{ep_idx:02d}_seed{ep_idx+args.base_seed}_f{k:02d}.png'
            frames[idx].save(path)
            frame_paths.append({'step': idx, 'path': path, 'qpos8': qpos_hist[idx]})
        return True, step, frame_paths
    return False, step, []


import json
print(f"Target: {args.n} success(es)  |  max attempts: {args.max_attempts}", flush=True)
print(f"\n{'ep':>4}  {'seed':>10}  {'result':>10}  {'steps':>6}  {'found':>6}", flush=True)
results, success_meta = [], []
ep = 0
while len(success_meta) < args.n and ep < args.max_attempts:
    s, t, frame_paths = rollout(ep)
    results.append(s)
    if s:
        success_meta.append({'ep': ep, 'seed': ep + args.base_seed,
                             'steps': t, 'frames': frame_paths})
    print(f"  {ep:2d}  {ep+args.base_seed:10d}    {'SUCCESS' if s else 'fail   '} ({t:4d})"
          f"  {len(success_meta)}/{args.n}", flush=True)
    ep += 1

if args.show:
    cv2.destroyAllWindows()

n_found = len(success_meta)
print(f"\nSuccess rate: {n_found}/{ep}  ({100*n_found/max(ep,1):.0f}%)", flush=True)
with open('success_frames.json', 'w') as f:
    json.dump({'task': args.task, 'base_seed': args.base_seed, 'successes': success_meta}, f)
print(f"Saved {n_found} success episodes ({args.frames} frames each) + success_frames.json", flush=True)
