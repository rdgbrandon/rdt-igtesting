"""Can ManiSkill render on this machine? Prints frame stats and step timings."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rdt_xai.rollout import _render, make_env  # noqa: E402

print("VK_ICD_FILENAMES =", os.environ.get("VK_ICD_FILENAMES"))
t0 = time.time()
env = make_env("PickCube-v1")
obs, _ = env.reset(seed=20241201)
print(f"env ready {time.time()-t0:.1f}s")
t0 = time.time()
img = _render(env)
print(f"first render {time.time()-t0:.2f}s size={img.size} mean={np.asarray(img).mean():.1f}")
t0 = time.time()
hold = np.r_[np.asarray(obs["agent"]["qpos"]).reshape(-1)[:7], 1.0].astype(np.float32)   # hold pose, gripper open
for _ in range(16):
    env.step(hold)
    _render(env)
print(f"16 steps+renders {time.time()-t0:.2f}s")
img.save("logs/render_check.png")
print("RENDER_OK")
