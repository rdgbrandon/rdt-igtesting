#!/usr/bin/env python3
"""
Pre-download all HuggingFace models into the local HF cache — independent of
the Colab notebook (replicates what Cells 2/4 download implicitly).

Idempotent: anything already cached is skipped instantly.

Usage:
    python local_download_models.py
"""
import os, shutil
from huggingface_hub import hf_hub_download, snapshot_download

BASE = os.path.dirname(os.path.abspath(__file__))
TASKS = ["PickCube-v1", "StackCube-v1", "PushCube-v1", "PegInsertionSide-v1", "PlugCharger-v1"]

print("1/4  Language embeddings ...")
for task in TASKS:
    dst = os.path.join(BASE, f"lang_embed_{task}.pt")
    if not os.path.exists(dst):
        shutil.copy(
            hf_hub_download("robotics-diffusion-transformer/maniskill-model",
                            "lang_embeds/text_embed_" + task + ".pt"),
            dst)
    print("   ", task, "OK")
lang_default = os.path.join(BASE, "lang_embed.pt")
if not os.path.exists(lang_default):
    shutil.copy(os.path.join(BASE, "lang_embed_PickCube-v1.pt"), lang_default)

print("2/4  SigLIP SO400M (~1.6 GB) ...")
snapshot_download("google/siglip-so400m-patch14-384",
                  allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors"])
print("    OK")

print("3/4  RDT-1B base (~5 GB) ...")
snapshot_download("robotics-diffusion-transformer/rdt-1b",
                  allow_patterns=["*.json", "*.txt", "*.md", "*.bin", "*.safetensors"])
print("    OK")

print("4/4  ManiSkill fine-tuned checkpoint ...")
hf_hub_download("robotics-diffusion-transformer/maniskill-model",
                "rdt/mp_rank_00_model_states.pt")
print("    OK")

print("\nAll models cached. Ready for local_rollout.py / local_ig.py.")
