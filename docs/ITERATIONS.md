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
| Panda gripper controller / SAPIEN renderer changed | diff shows the mimic gripper controller was rewritten; renderer 3.0.0b1 → 3.0.3 | **confirmed cause**: official eval on the Nov-2024 stack, n=50: **38/50 = 76%** (95% CI 63–86%; paper 77.2%) vs 10/24 = 42% on ManiSkill 3.0.1 (two-sided exact p=0.008) |

**Changed:** bf16 path fixed (timestep embedder dtype); rollouts back to official `obs_mode="rgb"`; RDT-era
environment builder; every stage runnable in either environment (`scripts/run_stage.py`, pilot `ENV` switch).

**Resolution:** RDT reproduces on ManiSkill 3.0.0b15 + SAPIEN 3.0.0b1 (Python 3.11 venv, `colab.rdt_era_env()`), not on
ManiSkill 3.0.1 / SAPIEN 3.0.3. All further RDT experiments use the RDT-era stack (pilot `ENV="rdt_era"`).
The newer-simulator sensitivity (76% → 42% with no change to policy or protocol) is itself worth a sentence in the paper.

**Confirmed (n=50, same seeds, RDT-era stack):** this repo **39/50 = 78%** (95% CI 65–87%) vs official 38/50 = 76%
(two-sided exact p=1.0), paper 77.2%. Successes take 57–146 steps; every failure times out at 400.

**Next:** rerun the pilot (`rdt_xai_pilot.ipynb`, `ENV="rdt_era"`) on the reproduced policy.

## Iteration 2 — pilot on the reproduced policy (Oct 27, 2026)

**Ran:** `rdt_xai_pilot.ipynb`, `ENV="rdt_era"`, output `pilot2`: 16 episodes/task, B 30 moments (fast curves),
C 15, D 6 episodes/condition at 10% grey patches, E 8, F 15.

**Showed:**
* **Policy reproduces:** PickCube 12/16 = 75%, StackCube 13/16 = 81%, PushCube 16/16 = 100%.
* **Denoising time (C), clean:** vision acts almost only in the last two of 5 steps: Expected-IG share 0.08 / 0.05 /
  0.06 / 0.23 / 0.59; plan change when vision is removed at one step 0.05 / 0.05 / 0.05 / 0.12 / 0.61.
* **Sanity (E), clean:** Spearman to the trained-model map after re-initializing 7 blocks: Expected IG 0.05,
  IG (bg) 0.07, BlurIG 0.08 vs DAAM 0.97, GAE 0.90, Gradient 0.88.
* **Faithfulness (B), not usable as is:** deletion AUC Expected IG 1.68 [0.97, 2.85], best mean but paired Wilcoxon
  significant only vs Random, ISS, Occlusion. Every method's deletion curve peaks at ~2.0-2.2 x the whole-view
  deviation after removing 5% of patches: grey squares are out of distribution, so the AUC mostly scores the
  artifact.
* **Closed loop (D), not usable as is:** 10% grey patches - top patches of every method 0/6; random already breaks
  StackCube (6/6 -> 2/6); no sign test significant.
* **Modality (F):** proprioception carries 53-71% of the Shapley share; the time plot had 2-3 points per task.
* `build_report` crashed on 0/6 conditions (Wilson bound rounding gave a negative error bar).

**Changed:**
* Patch removal is now **inpainted** from the remaining pixels (push-pull, `rdt_xai/fill.py`) in the metrics and
  the closed-loop test; full removal is still RDT's grey "missing camera" image. Stage B re-scores the saved maps
  into `B_inpaint/` (no re-explaining; `B/` keeps the grey scores) and the report compares the two fills.
* D: 5% of patches, 16 episodes, conditions none / random / eig top_abs / eig bottom_abs / eig top / DAAM top /
  BlurIG top, into `D_inpaint/`.
* F: modality Shapley at every decision of 8 successful episodes per task (`F_modality/`).
* Report: Wilson fix; one-cell notebook `rdt_xai_run_all.ipynb`; parallel Drive sync.

## Iteration 3 — inpainted removal, extended B/C/D/F

*(pending)*
