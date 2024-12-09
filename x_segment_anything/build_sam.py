# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
import os

import torch
from timm.models import create_model

from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer
from .modeling import TinyViT               # MobileSAM
from .modeling import RepViT                # EdgeSAM
from .modeling import repvit                # RepViTSAM
from .modeling import MaskDecoderCoralSCOP  # CoralSCOP

prompt_embed_dim = 256
image_size = 1024
vit_patch_size = 16
image_embedding_size = image_size // vit_patch_size


def build_sam_vit_h(checkpoint=None):
    image_encoder = _build_sam_encoder(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31]
    )
    return _build_sam(image_encoder, checkpoint)


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None):
    image_encoder = _build_sam_encoder(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23]
    )
    return _build_sam(image_encoder, checkpoint)


def build_sam_vit_b(checkpoint=None):
    image_encoder = _build_sam_encoder(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11]
    )
    return _build_sam(image_encoder, checkpoint)


# MobileSAM
def build_sam_vit_t(checkpoint=None):
    mobile_sam = Sam(
        image_encoder=TinyViT(img_size=1024, in_chans=3, num_classes=1000,
                              embed_dims=[64, 128, 160, 320],
                              depths=[2, 2, 6, 2],
                              num_heads=[2, 4, 5, 10],
                              window_sizes=[7, 7, 14, 7],
                              mlp_ratio=4.,
                              drop_rate=0.,
                              drop_path_rate=0.0,
                              use_checkpoint=False,
                              mbconv_expand_ratio=4.0,
                              local_conv_size=3,
                              layer_lr_decay=0.8
                              ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    mobile_sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, weights_only=True)
        mobile_sam.load_state_dict(state_dict)
    return mobile_sam


# EdgeSAM
def build_edge_sam(checkpoint=None, upsample_mode="bicubic"):
    image_encoder = RepViT(
        arch="m1",
        img_size=image_size,
        upsample_mode=upsample_mode
    )
    return _build_sam(image_encoder, checkpoint)


# RepViTSAM
def build_sam_repvit(checkpoint=None):
    repvit_sam = Sam(
        image_encoder=create_model('repvit'),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    repvit_sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, weights_only=True)
        repvit_sam.load_state_dict(state_dict)
    return repvit_sam


# CoralSCOP
def build_sam_vit_b_coralscop(checkpoint=None):
    return _build_sam_coralscop(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        cate_num=2,
        checkpoint=checkpoint,
    )


sam_model_registry = {
    "default": build_edge_sam,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
    "vit_t": build_sam_vit_t,
    "edge_sam": build_edge_sam,
    "repvit": partial(build_sam_repvit),
    "vit_b_coralscop": build_sam_vit_b_coralscop
}

sam_model_urls = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_t": "https://huggingface.co/spaces/dhkim2810/MobileSAM/resolve/main/mobile_sam.pt",
    "edge_sam_3x": "https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam_3x.pth",
    "edge_sam": "https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam.pth",
    "repvit": "https://huggingface.co/spaces/jameslahm/repvit-sam/resolve/main/repvit_sam.pt",
    "vit_b_coralscop": "https://github.com/Jordan-Pierce/CoralSCOP/releases/download/v0.0.1/vit_b_coralscop.pth"
}


# SAM
def _build_sam_encoder(
        encoder_embed_dim,
        encoder_depth,
        encoder_num_heads,
        encoder_global_attn_indexes,
):
    image_encoder = ImageEncoderViT(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
    )
    return image_encoder


def _build_sam(
        image_encoder,
        checkpoint=None,
):
    sam = Sam(
        image_encoder=image_encoder,
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location="cpu", weights_only=False)
        sam.load_state_dict(state_dict)
    return sam


# CoralSCOP
def _build_sam_coralscop(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    cate_num = 2,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoderCoralSCOP(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            cate_num=cate_num,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()

    if os.path.isfile(checkpoint):
        print("loading from "+checkpoint)
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict, strict=True)
    elif os.path.isdir(checkpoint):
        print("loading from " + checkpoint)
        with open(os.path.join(checkpoint,"image_encoder.pth"),"rb") as f_encoder:
            state_dict_encoder = torch.load(f_encoder)
        sam.image_encoder.load_state_dict(state_dict_encoder, strict=True)
        with open(os.path.join(checkpoint,"prompt_encoder.pth"),"rb") as f_prompt:
            state_dict_prompt = torch.load(f_prompt)
        sam.prompt_encoder.load_state_dict(state_dict_prompt, strict=True)

        with open(os.path.join(checkpoint,"mask_decoder.pth"),"rb") as f_decoder:
            state_dict_decoder = torch.load(f_decoder)
        sam.mask_decoder.load_state_dict(state_dict_decoder,strict=False)
    else:
        print("no checkpoint is provided!")

    return sam
