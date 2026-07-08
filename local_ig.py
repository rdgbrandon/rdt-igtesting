#!/usr/bin/env python3
"""
Local (Windows) IG analysis runner — independent of the Colab notebook.

Replicates notebook Cell 4: loads models via rdt_blurig.py with SKIP_IG=1, then
executes run_ig.py inside the same global namespace (the local equivalent of
IPython's `%run -i`). Does NOT modify rdt_blurig.py or run_ig.py.

Outputs land as PNGs in this directory (strip_ep*.png, ig_strip_ep*.png,
attr_temporal_ep*.png, joint_temporal_ep*.png, wj_ep*.png) — the inline
display() calls are no-ops outside a notebook.

Usage (after local_rollout.py has produced success_frames.json):
    python local_ig.py [--ddpm-steps 3]

--ddpm-steps: lower to 2-3 if CUDA runs out of memory during IG backprop on
8 GB cards. Fewer denoising steps = less activation memory retained per
backward pass; gradients get coarser but attribution maps stay usable.
"""
import os, sys, argparse, json, runpy

BASE = os.path.dirname(os.path.abspath(__file__))
RDT_REPO_DEFAULT = os.path.abspath(os.path.join(BASE, '..', 'RoboticsDiffusionTransformer'))

parser = argparse.ArgumentParser()
parser.add_argument('--task', default=None,
                    help='default: the task recorded in success_frames.json')
parser.add_argument('--ddpm-steps', type=int, default=None,
                    help='override diffusion steps for IG passes (VRAM knob)')
parser.add_argument('--rdt-repo', default=RDT_REPO_DEFAULT)
args = parser.parse_args()

os.chdir(BASE)

if args.task is None:
    with open('success_frames.json') as f:
        args.task = json.load(f)['task']
    print(f'LOCAL: task from success_frames.json: {args.task}')

os.environ.setdefault('DISPLAY', '')
os.environ['RDT_REPO']       = args.rdt_repo
os.environ['SKIP_IG']        = '1'
os.environ['MANISKILL_TASK'] = args.task
os.environ['LANG_EMBED']     = f'lang_embed_{args.task}.pt'
sys.path.insert(0, args.rdt_repo)

# ManiSkill's default GPU-direct render transfer (sapien_cuda) crashes with an
# access violation on Windows/WDDM. Wrap gym.make so the env rdt_blurig.py
# creates uses the CPU-copy render path — keeps rdt_blurig.py unmodified.
import gymnasium as _gymn
_orig_gym_make = _gymn.make
def _gym_make_cpu(*a, **kw):
    kw.setdefault('render_backend', 'cpu')
    return _orig_gym_make(*a, **kw)
_gymn.make = _gym_make_cpu

print('LOCAL: Loading models via rdt_blurig.py (SKIP_IG=1) ...')
_g = runpy.run_path(os.path.join(BASE, 'rdt_blurig.py'), run_name='__main__')

if args.ddpm_steps:
    rdt = _g['rdt']
    rdt.num_inference_timesteps = args.ddpm_steps
    for _attr in ('noise_scheduler_sample', 'noise_scheduler'):
        if hasattr(rdt, _attr):
            getattr(rdt, _attr).set_timesteps(args.ddpm_steps)
    _g['N_DDPM_STEPS'] = args.ddpm_steps
    print(f'LOCAL: DDPM steps for IG passes set to {args.ddpm_steps}')

print('LOCAL: Running run_ig.py in the loaded namespace ...')
_code = open(os.path.join(BASE, 'run_ig.py'), encoding='utf-8').read()
exec(compile(_code, 'run_ig.py', 'exec'), _g)
