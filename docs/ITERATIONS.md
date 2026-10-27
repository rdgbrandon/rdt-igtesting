# Iteration log

Each entry: what ran, what it showed, what changed. Numbers are copied from the runs, not estimated.

## Iteration 1 — pilot on Colab A100 (Oct 26–27, 2026)

**Ran:** `rdt_xai_pilot.ipynb` (ManiSkill 3.0.1, SAPIEN 3.0.3, Python 3.13, fp32). Measured speed: policy call
0.18 s, Expected IG (16 evals) 4.2 s, faithfulness for 11 methods 24 s; projected 2.8 h for the full pilot.

**Showed:**
* **Policy does not reproduce.** PickCube 1/6 = 17% (paper 77.2%), mean 359 of 400 steps; closed-loop control
  condition 1/8. A first repro run on StackCube (n=8): 4/8 with this pipeline (paper 74%).
* Attribution tables are therefore not usable (and n = 3 moments: CIs overlap, paired tests undefined).
* Only robust-looking signal: model-randomization sanity check — gradient / DAAM / GAE maps stay correlated
  with the trained model's (Spearman 0.6–0.98) after re-initializing RDT blocks; IG-family maps drop to ~0.
  To be re-checked on a working policy.

**Diagnosis so far:**
| Hypothesis | Test | Result |
|---|---|---|
| diffusers 0.42 scheduler differs from 0.27.2 | identical inputs through both `DPMSolverMultistepScheduler`s | identical timesteps and outputs → ruled out |
| our pipeline differs from RDT's `step()` | `tests/check_official_step.py`, real checkpoint, identical inputs + noise | max diff 4.8e-7 rad → ruled out |
| PickCube task / camera / frequencies changed | diff ManiSkill 3.0.0b15 vs 3.0.1 | same camera pose, spawn ranges, 100 Hz sim / 20 Hz control, `auto`→CPU sim |
| RDT's **official** eval on ManiSkill 3.0.1 | `colab.repro_check`, PickCube, n=10, official seeds | **4/10 = 40%** (paper 77.2%; one-sided exact p=0.015) → the gap is not specific to this repo |
| RDT-era stack (ManiSkill 3.0.0b15, SAPIEN 3.0.0b1, Py 3.11) | same check, n=10 | official 5/10, this repo 7/10; pooled 2024 sim 12/20 vs current 5/16 (incl. pilot 1/6), p=0.11 → **suggestive, not conclusive**; rerun at n=50 in a 2x2 design |
| more current-sim data | n=10 this repo 1/10; interrupted n=50 official run 10/24 (episodes 0–23) | current sim clearly below paper; every failure times out at 400 steps, successes take 59–152 steps |
| Panda gripper controller / SAPIEN renderer changed | diff shows the mimic gripper controller was rewritten; renderer 3.0.0b1 → 3.0.3 | **confirmed cause**: official eval on the Nov-2024 stack, n=50: **38/50 = 76%** (95% CI 63–86%; paper 77.2%) vs 10/24 = 42% on ManiSkill 3.0.1 (exact p≈0.006) |

**Changed:** bf16 path fixed (timestep embedder dtype); rollouts back to official `obs_mode="rgb"`; RDT-era
environment builder; every stage runnable in either environment (`scripts/run_stage.py`, pilot `ENV` switch).

**Resolution:** RDT reproduces on ManiSkill 3.0.0b15 + SAPIEN 3.0.0b1 (Python 3.11 venv, `colab.rdt_era_env()`), not on
ManiSkill 3.0.1 / SAPIEN 3.0.3. All further RDT experiments use the RDT-era stack (pilot `ENV="rdt_era"`).
The newer-simulator sensitivity (76% → 42% with no change to policy or protocol) is itself worth a sentence in the paper.

**Next:** confirm this repo's rollout at n=50 on the RDT-era stack, then rerun the pilot there.
