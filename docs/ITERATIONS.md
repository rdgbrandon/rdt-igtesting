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
  deviation after removing 5% of patches. Hypothesis: grey squares are out of distribution, so the AUC mostly
  scores the artifact. *(Refuted in iteration 3: the peak is the same with inpainting.)*
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

## Iteration 3 — inpainted removal, extended B/C/D/F (Oct 27–29, 2026)

**Ran:** `rdt_xai_run_all.ipynb` resuming `pilot2` (same rollouts, maps and E), A100, `ENV="rdt_era"`, two sessions
(the first disconnected in stage D; the rerun skipped finished work). B 60 moments (fast curves) re-scored with
inpainting, full curves; C 30; D 10 episodes per condition, 5% of patches, inpainted; F 30 moments + modality at
every decision of 8 successful episodes per task. Roughly 2.5 h of GPU time over both sessions.

**Showed:**
* **Inpainting did not remove the deletion-curve overshoot:** mean peak D/D_full 1.94-2.18 for every method
  except ISS (1.72) and Grad x Input (1.40) (grey fill: 1.89-2.04; Random 1.21 vs 1.11). So it is not a grey-square artifact.
  Likely reading: with no camera RDT falls back on joint state (64-70% Shapley share) and its plan stays close,
  while a view with the object hidden is a different scene. D_full is shared within a moment, so per-moment
  comparisons are unaffected, but means are not: ISS scored 16.2 on one moment with a tiny D_full.
* **Faithfulness (B) does not favour Expected IG.** Inpainted deletion (mean): BlurIG 1.19, Occlusion 1.18,
  IG (bg) 1.15, Expected IG 1.14, Gradient 1.13, ... Random 0.72; Expected IG vs every non-trivial baseline n.s.
  except Grad x Input (p=4e-4) and ISS (p=7e-6). Insertion: DAAM 0.57, Gradient 0.56, GAE 0.53 beat Expected IG
  0.38 (Wilcoxon p<2e-6). Deletion ranking depends on the fill: Expected IG 1st with grey, 4th inpainted.
* **Closed loop (D) is causal for Expected IG but does not separate methods.** Success with 5% hidden,
  PickCube / StackCube: none 7/10, 9/10; random 4/10, 6/10; eig top_abs 0/10, 0/10; eig bottom_abs 6/10, 10/10
  (bottom_abs vs top_abs sign test p=0.031, 0.002; random vs top_abs StackCube p=0.031). DAAM top 0/10, 0/10 and
  BlurIG top 2/10, 0/10: floor effect at 5%.
* **Denoising time (C), n=30, holds:** Expected-IG share by step 0.08 / 0.05 / 0.07 / 0.21 / 0.60; plan change with
  vision removed at one step 0.04 / 0.04 / 0.05 / 0.09 / 0.57; plan recovered with vision at one step only
  0.09 / 0.07 / 0.14 / 0.39 / 0.81.
* **Modality (F):** over every decision of whole episodes, vision 0.33 / 0.30 / 0.24, language 0.03 / 0.05 / 0.06,
  joint state 0.64 / 0.65 / 0.70 (PickCube / PushCube / StackCube). Word IG vs word removal rank agreement
  0.44-0.57.
* E unchanged from iteration 2 (Expected IG, IG, BlurIG ~0; DAAM 0.97, GAE 0.90, Gradient 0.88 at 7 blocks).

**Changed:** B is summarized with medians, mean per-moment ranks and median paired differences; moments with
D_full below 0.25 x the median are flagged and means are shown with and without them; the report no longer calls
D/D_full > 1 an artifact.

**Open:** separate methods in D (1-2% of patches, more episodes); randomize SigLIP too in E (only RDT blocks are
re-initialized, so maps that track SigLIP's input sensitivity can survive); paper framing — "faithful" in the
causal sense (D) and the model-dependence sense (E), while deletion / insertion rankings are fill-dependent and
disagree with E.

## Iteration 4 — object grounding (stage G)

**Added:** `rdt_xai/grounding.py`. Each saved episode is replayed (seed + recorded actions, replayed RGB checked
against the saved frames) to get the simulator's per-pixel object ids of the frames RDT saw. Per moment and method:
attribution share per object (task objects, gripper, arm, background), and the plan change when that object is
removed from both frames (inpainted) vs an equally large set of background patches. Scores: rank agreement between
claimed share and measured effect, top-1 agreement, task-object enrichment, and the split before / after the grasp.

*(results pending)*

## Iteration 5 — second policy: ManiSkill's Diffusion Policy baseline (Oct 29, 2026)

**Setup:** `colab.train_dp` runs ManiSkill 3.0.1's `train_rgbd.py` (RGB, 128x128 base camera, 2 frames, U-Net,
100 DDPM steps, pd_ee_delta_pos, 100 motion-planning demos, 30k iterations, settings of its `baselines.sh`) on the
Colab stack: gymnasium 1.4.0, diffusers 0.42, torch 2.11, numpy 2.1, Python 3.13.

**Three incompatibilities of the baseline with that stack** (patched by `DP_PATCHES`, nothing in ManiSkill itself):
* eval env workers started with `forkserver` hit "Cannot re-initialize CUDA in forked subprocess": use `spawn`;
* gymnasium >= 1.0 resets a finished vector env one step late and drops `final_info`: same-step autoreset, and read
  the ending step's infos as a dict of arrays;
* **silent:** gymnasium >= 1.0's `batch_space` sorts Dict keys, so the training data (reordered to the env's batched
  observation space) packs the state alphabetically (extra: goal_pos, is_grasped, tcp_pose), while the eval wrapper
  packs the env's own order (is_grasped, tcp_pose, goal_pos). The evaluated policy got the goal where it learned the
  gripper pose. PickCube at 25k iterations: 0/100 in the baseline's eval, while it fit the demo actions to a mean
  error of 0.003 (0.302 for the mean action), open-loop demo replay succeeded, and the live and demo images matched
  (mean diff 0.5/255). Fix: flatten live obs in the batched space's order (`SpaceOrderFlatten`). `dp_diagnose`
  runs these checks.

**Result:** PickCube, same 25k checkpoint, fixed eval, 100 fresh episodes: success_once 0.91, success_at_end 0.62.
PushCube: *(training)*.

**Caveat for any DP number:** a policy trained before the fix is fine; only its in-training eval scores (and so
`best_eval_*.pt`) are wrong. The attribution adapter must build the state in the same sorted order.

**Adapter (`rdt_xai/dp.py`, `rdt_xai/dp_stages.py`, `scripts/run_dp.py`, `notebooks/dp_xai.ipynb`):** the study's
stages on the DP checkpoint, in pixel space (2 frames x 128x128, 16x16 grid of 8-px patches). No-vision input =
mean frame of the stage-A episodes (DP never saw a missing camera); 100 DDPM steps in 5 bins of 20 for C; DAAM /
GAE replaced by Grad-CAM (CNN encoder); G uses the sensor camera's own segmentation, recorded live (no replay).
`tests/test_dp.py` (random Agent, fake env, CPU): the written-out DDPM update equals diffusers' step exactly,
conditioning equals `Agent.encode_obs`, EIG completeness error 0.3%, bins sum to EIG, full removal = no-vision
image, all stages run.
