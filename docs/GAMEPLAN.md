# Game plan: faithful attribution for diffusion robot policies (CVPR 2027)

**Working title.** *Where — and when — does a diffusion policy look? Faithful visual attribution for
vision-language-action diffusion transformers.*

**Deadlines.** CVPR 2027: paper registration **Nov 10, 2026**, submission **Nov 16, 2026** (AoE),
supplementary Nov 23. Fallbacks if the pilot is weak: ICCV 2027 / CVPR workshops (spring 2027).

## 1. What was wrong with the current pipeline

| Issue in `main` | Effect | Fix in `cvpr-xai` |
|---|---|---|
| `run_ig.py` builds the language mask from *shortened* instructions | 10/30 StackCube tokens (6/25 Peg, 6/24 Plug) were hidden from the model; the "pad" columns in the report are real words | Released `.pt` embeddings are unpadded → all-ones mask, official instructions |
| Attribution feeds the same frame to all 6 camera slots | Explains an input the policy never sees (rollouts use `[prev, bg, bg, curr, bg, bg]`) | Attribute the two external slots jointly, wrist slots stay at the real background |
| Score = norm of absolute joint positions | Dominated by the current pose, not the decision | Signed projection of the executed plan onto (plan − plan without vision); exact completeness |
| One fixed sampler seed | Attribution of one noise draw of a stochastic policy | Expected IG over sampler noise at the same cost |
| Report says `acts[::4]` is a bug | It is the official protocol (64 interpolated steps → 16 executed) | Keep stride 4, fix the report |
| No quantitative evaluation | Heatmaps cannot be compared or trusted | Pixel-space deletion/insertion on held-out noise, sanity checks, closed-loop masking |

## 2. Positioning against prior work

* **Attribution:** IG (Sundararajan '17), BlurIG (Xu '20), Expected Gradients (Erion '21), SmoothGrad,
  Grad-CAM, occlusion (Zeiler '14), RISE (Petsiuk '18); evaluation via deletion/insertion (Petsiuk '18),
  ROAR/ROAD, sanity checks (Adebayo '18), baseline choice (Sturmfels '20).
* **Attention as explanation:** Jain & Wallace '19, attention rollout (Abnar '20), Chefer '21 (GAE, gradient-
  weighted cross-attention), DAAM (Tang '23: cross-attention aggregated over denoising steps in Stable Diffusion).
* **Robotics / VLAs:** causal confusion (de Haan '19); attention maps of OpenVLA / π0 are diffuse
  (Don't Blind Your VLA '25, Look Before Acting '26, VLA-Trace '26). Closest: *Embodied Interpretability*
  (ICML '26) — gradient-free random masking (ISS) on π0.5, offline only, compared only to attention/token-norm,
  no treatment of sampler noise, no closed-loop masking.
* **Gap we fill:** gradient-based, diffusion-aware attribution for a diffusion VLA (RDT-1B), compared against
  attention (DAAM, GAE) and perturbation (occlusion, ISS) families, validated **in closed loop**, plus an analysis
  of **when during denoising** vision is used.

## 3. Contributions (what the paper claims if the pilot holds)

1. **Policy-faithful attribution protocol** for diffusion action policies: executed-chunk projection target
   with exact completeness; RDT's native "missing camera" image as baseline; **Expected IG over sampler noise**
   (same cost as IG); joint attribution over history frames.
2. **Denoising-time attribution:** an *exact* split of expected IG by denoising step (each step gets its own
   copy of the input; per-step maps sum to the total), plus single-step vision knockout.
3. **Benchmark:** 11 methods at matched compute on RDT-1B × ManiSkill3, pixel-space faithfulness on held-out
   noise, seed stability, model-randomization sanity checks.
4. **Closed-loop causal test:** hiding the top-attributed 10% of patches *while the robot acts* vs random /
   bottom-10% — task success as the faithfulness metric.

## 4. Hypotheses the pilot tests

| | Hypothesis | Evidence in the notebook |
|---|---|---|
| H1 | Single-seed attributions of a diffusion policy are unstable across sampler noise; Expected IG is more stable | `seed_spearman` column (B) |
| H2 | Expected IG beats IG-zero / BlurIG / gradient on deletion & insertion | B table + paired Wilcoxon |
| H3 | Raw cross-attention (DAAM) is less faithful than gradient-weighted attention (GAE) and path methods | B table |
| H4 | Hiding Expected-IG top patches hurts success far more than random; bottom patches barely matter | D figure + sign tests |
| H5 | Vision matters unevenly over denoising steps (concentrated in a few steps) | C figure |
| H6 | Both history frames carry attribution (single-slot attribution under-counts) | `prev_share` column |

Negative outcomes are still publishable as a benchmark (e.g. "occlusion/ISS are most faithful but 5× costlier;
attention is unreliable for diffusion VLAs"), so the paper does not hinge on our method winning everything.

## 5. Schedule (manageable, 3 weeks)

| Dates | Work | Compute |
|---|---|---|
| Oct 27–28 | **Pilot notebook** (`notebooks/rdt_xai_pilot.ipynb`): 3 tasks × 16 episodes, 60 moments, closed loop 2 tasks × 6 conditions × 10 episodes | ~3–4 A100 h |
| Oct 29–Nov 2 | Fix whatever the pilot shows; scale to 5 tasks × 50 episodes, ~300 moments, closed loop 25 episodes/condition | ~15–20 A100 h (Colab) or Expanse array jobs |
| Nov 3–6 | Extensions if time: second policy (Diffusion Policy checkpoint shipped in the same HF repo), success-vs-failure attribution, per-joint maps | ~5–10 A100 h |
| Nov 7–10 | Figures, related work, method section; **register paper Nov 10** | — |
| Nov 11–16 | Writing, ablations reviewers will ask for (n_evals, #seeds, mask fraction), submit | small |

## 6. Risks and what we do about them

* **RDT barely uses vision on some tasks** (plan dominated by proprioception): `d_none` in C measures the
  vision effect size — report it; it is itself a finding (causal confusion).
* **Success rates do not reproduce** the paper (77/74/100%): check ManiSkill version & render camera before
  anything else — stage A prints the comparison.
* **Expected IG does not win:** the benchmark + closed-loop + denoising-time analysis still stand; frame the
  method as the diffusion-aware fix to IG and report honestly.
* **Compute:** fast preset (`fast=True`) is ~3× cheaper; Expanse CPU (30K core-h) is the overflow.
