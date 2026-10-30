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
PushCube, 25k checkpoint, same eval: success_once 0.86, success_at_end 0.86 (100 episodes). Both checkpoints are
25k-iteration EMA weights (numbered saves stop at the last multiple of 5000 below 30k).

**Caveat for any DP number:** a policy trained before the fix is fine; only its in-training eval scores (and so
`best_eval_*.pt`) are wrong. The attribution adapter must build the state in the same sorted order.

**Adapter (`rdt_xai/dp.py`, `rdt_xai/dp_stages.py`, `scripts/run_dp.py`, `notebooks/dp_xai.ipynb`):** the study's
stages on the DP checkpoint, in pixel space (2 frames x 128x128, 16x16 grid of 8-px patches). No-vision input =
mean frame of the stage-A episodes (DP never saw a missing camera); 100 DDPM steps in 5 bins of 20 for C; DAAM /
GAE replaced by Grad-CAM (CNN encoder); G uses the sensor camera's own segmentation, recorded live (no replay).
`tests/test_dp.py` (random Agent, fake env, CPU): the written-out DDPM update equals diffusers' step exactly,
conditioning equals `Agent.encode_obs`, EIG completeness error 0.3%, bins sum to EIG, full removal = no-vision
image, all stages run.

## Iteration 6 — the study on the Diffusion Policy (Oct 30, 2026)

`notebooks/dp_xai.ipynb`, defaults (A 20 episodes, B 40 moments, C 30, D 16 episodes x 7 conditions at 5%, E 10
moments, F 8 episodes). **PickCube** (complete):
* **A:** 19/20.
* **C (late denoising replicates, more sharply than RDT):** EIG share over 5 bins of 20 DDPM steps (noisy -> clean)
  0.00 / 0.01 / 0.03 / 0.09 / 0.87; plan change with vision removed in one bin 0.01 / 0.01 / 0.01 / 0.03 / 0.79;
  recovered with vision in that bin only 0.05 / 0.13 / 0.06 / 0.09 / 0.85. PushCube (C done): every moment
  0 / 0 / 0 / 0 / 1.0.
* **D (causal, floor effect at 5%):** none 13/16, random 11/16, eig bottom_abs 12/16; eig top_abs, eig top, Grad-CAM
  top, BlurIG top all 0/16. PushCube: none 9/16, random 10/16, bottom_abs 9/16 (the same episodes as none),
  top_abs 2/16, top 2/16.
* **B:** every method far above random (median deletion 0.74-0.93 vs 0.43), small spread among methods; mean
  deletion rank: Gradient 3.5, EIG 3.7, IG (bg) 3.8, BlurIG 4.8, Grad-CAM 5.1, IG (zero) 5.5, Occlusion 5.5,
  Grad x Input 6.0, ISS 7.6, random 9.4.
* **G:** removing the arm 0.57 (control 0.04), cube 0.35 (0.01), gripper 0.25 (0.01): the arm matters most
  (cf. visual embodiment dependence, arXiv 2609.16815). Claim vs effect Spearman (3 groups, coarse): EIG / BlurIG
  0.46, IG (bg) 0.45, Grad x Input 0.39, IG (zero) 0.34; Gradient 0.03, Occlusion 0.00, ISS 0.03, Grad-CAM -0.05.
  Gradient ranks first on B but last on G.
* **E (EIG fails here):** map correlation with the trained model's after randomizing everything incl. the encoder:
  Grad-CAM 0.34, IG (zero) 0.36, BlurIG 0.38, Occlusion 0.41, Gradient 0.47, Grad x Input 0.67, IG (bg) 0.68,
  EIG 0.71, ISS 0.80. U-Net-only levels change little for any method (pixel maps pass through the intact
  max-pooled encoder). Likely reason for EIG / IG (bg) / Grad x Input: their (x - mean frame) factor, which marks
  whatever moved (arm, cube) whatever the model. RDT's E randomized only RDT blocks on SigLIP embeddings, so the
  two E results are not comparable until SigLIP is randomized too.
* **F:** vision 0.21, proprioception 0.79 (RDT 0.24-0.33 vision).

**Open (from this):** (1) is "late" specific to vision? knock proprioception out per bin as a control; (2) G may
inherit the (x - baseline) confound: score G with the randomized model's maps as a floor; (3) D at 1-2% to separate
methods; (4) SigLIP randomization for RDT's E.
