"""CPU test of the Diffusion Policy adapter (dp.py, dp_stages.py) on a randomly initialized baseline
Agent and a fake ManiSkill env. Needs the (patched) baseline code: DP_CODE=<.../diffusion_policy>.

  DP_CODE=... python tests/test_dp.py
"""
import glob
import json
import os
import sys
import tempfile
import types

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rdt_xai import dp as D  # noqa: E402
from rdt_xai import dp_stages as S  # noqa: E402

FAILS = []


def check(name, ok, detail=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if detail is None else f"  {detail}"), flush=True)
    if not ok:
        FAILS.append(name)


torch.manual_seed(0)
tmp = tempfile.mkdtemp()
task = "PickCube-v1"

# a random baseline Agent, saved as the training script saves it
sys.path.insert(0, D.DP_CODE)
import train_rgbd as T  # noqa: E402
from gymnasium import spaces  # noqa: E402

STATE_DIM, ACT_DIM = 29, 4
env_spaces = types.SimpleNamespace(
    single_observation_space=spaces.Dict(dict(state=spaces.Box(-9, 9, (2, STATE_DIM), np.float32),
                                              rgb=spaces.Box(0, 255, (2, 128, 128, 3), np.uint8))),
    single_action_space=spaces.Box(-1, 1, (ACT_DIM,), np.float32))
agent = T.Agent(env_spaces, T.Args())
for prm in agent.parameters():                       # make the random policy depend on its inputs
    prm.data += 0.02 * torch.randn_like(prm)
ckpt = os.path.join(tmp, "final.pt")
torch.save(dict(agent=agent.state_dict(), ema_agent=agent.state_dict()), ckpt)
ag = D.load_agent(ckpt, device="cpu")
check("load_agent dims", ag.act_dim == ACT_DIM and ag.visual_encoder is not None
      and ag.noise_pred_net.final_conv[1].weight.shape[0] == ACT_DIM)
pol = D.DPPolicy(ag)

# the written-out DDPM update equals diffusers' DDPMScheduler.step with the same noises
cond = pol.conditions(np.random.default_rng(0).integers(0, 255, (2, 128, 128, 3), dtype=np.uint8),
                      np.random.default_rng(1).normal(size=(2, STATE_DIM)).astype(np.float32))
gen = lambda i: torch.Generator().manual_seed(1000 + i)
z0 = torch.randn(1, 16, ACT_DIM, generator=torch.Generator().manual_seed(7))
zs = torch.stack([torch.randn(1, 16, ACT_DIM, generator=gen(i)) for i in range(pol.T)])
with torch.no_grad():
    mine = pol.sample(cond, noise=(z0, zs))
    obs_cond = pol._cond(pol.features(cond["x"]), cond["state"])
    a = z0.clone()
    for i, t in enumerate(ag.noise_scheduler.timesteps):
        eps = ag.noise_pred_net(sample=a, timestep=t, global_cond=obs_cond)
        a = ag.noise_scheduler.step(model_output=eps, timestep=t, sample=a, generator=gen(i)).prev_sample
check("DDPM update matches diffusers", torch.allclose(mine, a, atol=1e-5), f"max diff {(mine - a).abs().max():.2e}")
with torch.no_grad():
    enc = ag.encode_obs(dict(rgb=cond["x"] * 255, state=cond["state"]), eval_mode=True)
check("conditioning matches Agent.encode_obs", torch.allclose(enc, obs_cond, atol=1e-5))
check("seeded sampling is deterministic", torch.equal(pol.sample(cond, seed=3), pol.sample(cond, seed=3)))


# a fake env: a red cube drifts across a grey table; obs as ManiSkill returns them (batch 1)
class FakeEnv:
    def __init__(self):
        sp = lambda n: spaces.Box(-9, 9, (1, n), np.float32)
        self.unwrapped = self
        self.observation_space = spaces.Dict(dict(           # plain dicts: gymnasium sorts the keys
            agent=spaces.Dict(dict(qpos=sp(9), qvel=sp(9))),
            extra=spaces.Dict(dict(is_grasped=sp(1), tcp_pose=sp(7), goal_pos=sp(3)))))
        self.segmentation_id_map = {1: types.SimpleNamespace(name="cube"), 2: types.SimpleNamespace(name="goal_site"),
                                    3: types.SimpleNamespace(name="panda_hand"), 4: types.SimpleNamespace(name="panda_link3")}

    def reset(self, seed=None):
        self.t, self.rng = 0, np.random.default_rng(seed)
        return self._obs(), {}

    def _obs(self):
        rgb = np.full((128, 128, 3), 120, np.uint8)
        seg = np.zeros((128, 128), np.int16)
        c = 20 + 3 * self.t
        rgb[60:76, c:c + 16] = (200, 30, 30); seg[60:76, c:c + 16] = 1
        rgb[20:30, 100:110] = (30, 200, 30); seg[20:30, 100:110] = 2
        rgb[0:20, 40:70] = (90, 90, 90); seg[0:20, 40:70] = 3
        rgb[0:10, 70:128] = (60, 60, 60); seg[0:10, 70:128] = 4
        to = lambda v: torch.as_tensor(np.asarray(v, np.float32))[None]
        return dict(agent=dict(qpos=to(np.full(9, 0.1 * self.t)), qvel=to(np.zeros(9))),
                    extra=dict(is_grasped=torch.tensor([self.t > 20]), tcp_pose=to(np.arange(7)),
                               goal_pos=to([7, 8, 9])),
                    sensor_param=dict(base_camera=dict(k=torch.zeros(1, 3, 3))),
                    sensor_data=dict(base_camera=dict(rgb=torch.from_numpy(rgb)[None],
                                                      segmentation=torch.from_numpy(seg)[None, ..., None])))

    def step(self, a):
        self.t += 1
        return self._obs(), 0.0, False, False, dict(success=torch.tensor([self.t >= 20 + self.rng.integers(0, 9)]),
                                                     is_grasped=torch.tensor([self.t > 20]))

    def close(self):
        pass


fe = FakeEnv()
obs, _ = fe.reset(seed=0)
sv = D.state_vector(obs, fe.observation_space)
check("state packed in sorted key order (goal_pos, is_grasped, tcp_pose)",
      sv.shape == (STATE_DIM,) and np.allclose(sv[18:21], [7, 8, 9]) and np.allclose(sv[22:29], np.arange(7)), sv[18:])

D.make_env = lambda task, segmentation=True, max_steps=None: FakeEnv()
out = os.path.join(tmp, "out", task)
S.rollouts(pol, task, range(3), os.path.join(out, "A"), verbose=False)
meta = D.load_episode(glob.glob(os.path.join(out, "A", f"{task}_ep000/"))[0])
check("A: rollout recorded", meta["n_calls"] >= 2 and meta["rgb"].shape[1:] == (128, 128, 3)
      and meta["seg"].shape[1:] == (128, 128) and meta["calls"][1]["frames"] == [7, 8]
      and meta["seg_names"][1] == "cube", f"calls={meta['n_calls']} frames={meta['calls'][1]['frames']}")
pol.set_background(*S.background(os.path.join(out, "A"), task))
check("A: background", pol.bg.shape == (3, 128, 128) and pol.state_mean.shape == (STATE_DIM,))

ex = D.DPExplainer(pol, batch=4)
frames, states, seg = D.moment(meta, 1)
cond = pol.conditions(frames, states)
for m in ("ig_bg", "eig"):
    r = ex.explain(cond, m, n_evals=8, n_seeds=2, seed=1)
    check(f"{m} completeness", r["completeness_rel_err"] < 0.35, f"rel err {r['completeness_rel_err']:.3f}")
r = ex.explain(cond, "eig_steps", n_evals=8, n_seeds=2, seed=1)
e = ex.explain(cond, "eig", n_evals=8, n_seeds=2, seed=1)
check("eig_steps sums to eig", r["steps"].shape == (5, 16, 16) and np.allclose(r["steps"].sum(0), e["map"], atol=1e-4))
for m in D.METHODS:
    r = ex.explain(cond, m, n_evals=4, n_seeds=2, seed=1)
    check(f"method {m}", r["map"].shape == (16, 16) and np.isfinite(r["map"]).all() and np.abs(r["map"]).sum() > 0,
          f"{r['seconds']:.1f}s")
ko = ex.knockout(cond, seed=1)
check("knockout", ko["drop"].shape == (5,) and np.isfinite(ko["drop"]).all(), np.round(ko["drop"], 2))
mo = ex.modality(cond)
check("modality", abs(mo["vision"] + mo["state"] - 1) < 1e-6, f"vision {mo['vision']:.2f}")

fm = D.DPFaithfulness.fast(pol, batch=8)
x_all = fm.remove(cond["x"], D.patch_pixels(np.ones(256, bool)))
check("removing every patch gives the no-vision image", torch.allclose(x_all[0], pol.bg.expand(2, -1, -1, -1), atol=1e-5))

S.benchmark(pol, task, S.moments(os.path.join(out, "A"), task, only_success=False)[:2], ["random", "grad", "eig"],
            os.path.join(out, "B"), n_evals=4, fast=True, verbose=False)
B = [json.load(open(f)) for f in glob.glob(os.path.join(out, "B", "*.json"))]
g = B[0]["grounding"]
check("B+G", len(B) == 2 and 0 <= B[0]["methods"]["eig"]["deletion_morf"] < 10 and g["groups"][:2] == ["cube", "goal"]
      and np.isfinite(g["effect"][0]) and abs(sum(g["area_share"]) - 1) < 1e-6, g["effect"])
S.denoise(pol, S.moments(os.path.join(out, "A"), task, only_success=False)[:1], os.path.join(out, "C"), n_evals=4,
          verbose=False)
S.closed_loop(pol, task, range(2), ["none:none", "random:random", "eig:top_abs"], os.path.join(out, "D"), frac=0.05,
              n_evals=2, verbose=False)
check("D", len(glob.glob(os.path.join(out, "D", "*.json"))) == 6)
S.sanity(pol, S.moments(os.path.join(out, "A"), task, only_success=False)[:1], ["grad", "random"],
         os.path.join(out, "E"), n_evals=2, verbose=False)
E = json.load(open(os.path.join(out, "E", "sanity.json")))
check("E", len(E) == 10 and np.isfinite([r["spearman_abs"] for r in E if r["method"] == "grad"]).all())
S.modality(pol, os.path.join(out, "A"), task, 1, os.path.join(out, "F"), verbose=False)
txt = S.summary(out, task)
print(txt)
check("summary", all(s in txt for s in ("A  success", "B  faithfulness", "G  removing", "C  over", "D  closed",
                                         "E  map rank", "F  over")))
print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
