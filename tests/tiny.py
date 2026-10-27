"""Tiny random RDT + SigLIP with the real architecture and input layout, for fast CPU tests."""
import torch
from transformers import SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel

from rdt_xai.policy import RDTPolicy, RDTRunner, default_paths


def tiny_policy():
    torch.manual_seed(0)
    vcfg = SiglipVisionConfig(image_size=384, patch_size=14, hidden_size=32, intermediate_size=64,
                              num_hidden_layers=1, num_attention_heads=2)
    vision = SiglipVisionModel(vcfg)
    proc = SiglipImageProcessor.from_pretrained(default_paths()["siglip"])
    cfg = dict(lang_adaptor="mlp2x_gelu", img_adaptor="mlp2x_gelu", state_adaptor="mlp3x_gelu",
               rdt=dict(hidden_size=64, depth=4, num_heads=4, cond_pos_embed_type="multimodal"),
               noise_scheduler=dict(type="ddpm", num_train_timesteps=1000, num_inference_timesteps=5,
                                    beta_schedule="squaredcos_cap_v2", prediction_type="sample", clip_sample=False))
    rdt = RDTRunner(action_dim=128, pred_horizon=64, config=cfg, lang_token_dim=4096, img_token_dim=32,
                    state_token_dim=128, max_lang_cond_len=1024, img_cond_len=6 * 729,
                    img_pos_embed_config=[("image", (2, 3, -729))], lang_pos_embed_config=[("lang", -1024)],
                    dtype=torch.float32)
    torch.nn.init.normal_(rdt.model.final_layer.ffn_final.fc2.weight, std=0.5)   # RDT zero-inits it
    return RDTPolicy(model=rdt, vision=vision, processor=proc, dtype=torch.float32)
