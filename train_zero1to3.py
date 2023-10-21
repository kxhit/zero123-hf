#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from dataset import ObjaverseDataLoader, ObjaverseData
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import PretrainedConfig, CLIPVisionModelWithProjection, CLIPImageProcessor, CLIPFeatureExtractor

import diffusers
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    UNet2DConditionModel,
)
from pipeline_zero1to3 import Zero1to3StableDiffusionPipeline, CCProjection
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.training_utils import EMAModel
import torchvision
import kornia
import itertools

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.19.0.dev0")

logger = get_logger(__name__)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def log_validation(validation_dataloader, vae, image_encoder, feature_extractor, unet, cc_projection, args, accelerator, weight_dtype, split="val"):
    logger.info("Running {} validation... ".format(split))

    scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    pipeline = Zero1to3StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae).eval(),
        image_encoder=accelerator.unwrap_model(image_encoder).eval(),
        feature_extractor=feature_extractor,
        unet=accelerator.unwrap_model(unet).eval(),
        cc_projection=accelerator.unwrap_model(cc_projection).eval(),
        scheduler=scheduler,
        safety_checker=None,
        torch_dtype=weight_dtype,
    )

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    image_logs = []
    for valid_step, batch in tqdm(enumerate(validation_dataloader)):
        if args.num_validation_batches is not None and valid_step >= args.num_validation_batches:
            break
        gt_image = batch["image_target"].to(dtype=weight_dtype)
        input_image = batch["image_cond"].to(dtype=weight_dtype)
        pose = batch["T"].to(dtype=weight_dtype)
        images = []
        h, w = input_image.shape[2:]
        for _ in range(args.num_validation_images):
            with torch.autocast("cuda"):
                image = pipeline(input_imgs=input_image, prompt_imgs=input_image, poses=pose, height=h, width=w,
                                 guidance_scale=args.guidance_scale, num_inference_steps=50, generator=generator).images[0]

            images.append(image)

        image_logs.append(
            {"gt_image": gt_image, "pred_images": images, "pose": pose, "input_image": input_image}
        )

    # after validation, set the pipeline back to training mode
    unet.train()
    vae.train()
    image_encoder.train()
    cc_projection.train()


    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            formatted_images = []

            for log_id, log in enumerate(image_logs):
                pred_images = log["pred_images"]  # pred
                input_image = log["input_image"]    # input
                gt_image = log["gt_image"]  # GT

                formatted_images.append(wandb.Image(input_image, caption="{}_input".format(log_id)))
                formatted_images.append(wandb.Image(gt_image, caption="{}_gt".format(log_id)))

                for sample_id, pred_image in enumerate(pred_images): # n_samples
                    pred_image = wandb.Image(pred_image, caption="{}_pred_{}".format(log_id, sample_id))
                    formatted_images.append(pred_image)

            tracker.log({split: formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    # del pipeline
    # torch.cuda.empty_cache()

    return image_logs


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_input.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- diffusers
inference: true
---
    """
    model_card = f"""
# zero123-{repo_id}

These are zero123 weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a Zero123 training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="lambdalabs/sd-image-variations-diffusers",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="zero123-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=200000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.0,
        help="unconditional guidance scale, if guidance_scale>1.0, do_classifier_free_guidance"
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=0.05,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=5000,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=1,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=100, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",    # log_image currently only for wandb
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", default=True, help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        default=True,
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")

    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=2,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--num_validation_batches",
        type=int,
        default=8,
        help=(
            "Number of batches to use for validation. If `None`, use all batches."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_zero123_hf",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images."
        )

    return args

def CLIP_preprocess(x):
    dtype = x.dtype
    if isinstance(x, torch.Tensor):
        if x.min() < -1.0 or x.max() > 1.0:
            raise ValueError("Expected input tensor to have values in the range [-1, 1]")
    x = kornia.geometry.resize(x.to(torch.float32), (224, 224), interpolation='bicubic', align_corners=True, antialias=False).to(dtype=dtype)   # not bf16
    x = (x + 1.) / 2.
    # renormalize according to clip
    x = kornia.enhance.normalize(x, torch.Tensor([0.48145466, 0.4578275, 0.40821073]),
                                 torch.Tensor([0.26862954, 0.26130258, 0.27577711]))
    return x

def _encode_image(image_encoder, image, device, dtype, do_classifier_free_guidance):

    image = image.to(device=device, dtype=dtype)
    image = CLIP_preprocess(image)
    # if not isinstance(image, torch.Tensor):
    #     # 0-255
    #     print("Warning: image is processed by hf's preprocess, which is different from openai original's.")
    #     image = self.feature_extractor(images=image, return_tensors="pt").pixel_values
    image_embeddings = image_encoder(image).image_embeds.to(dtype=dtype)
    image_embeddings = image_embeddings.unsqueeze(1)

    if do_classifier_free_guidance:
        negative_prompt_embeds = torch.zeros_like(image_embeddings)
        image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])

    return image_embeddings.detach()

def _encode_pose(pose, device, dtype, do_classifier_free_guidance):
    if isinstance(pose, torch.Tensor):
        pose_embeddings = pose.unsqueeze(1).to(device=device, dtype=dtype)
    else:
        if isinstance(pose[0], list):
            pose = torch.Tensor(pose)
        else:
            pose = torch.Tensor([pose])
        x, y, z = pose[:, 0].unsqueeze(1), pose[:, 1].unsqueeze(1), pose[:, 2].unsqueeze(1)
        pose_embeddings = torch.cat([torch.deg2rad(x),
                                     torch.sin(torch.deg2rad(y)),
                                     torch.cos(torch.deg2rad(y)),
                                     z], dim=-1).unsqueeze(1).to(device=device, dtype=dtype)  # B, 1, 4

    if do_classifier_free_guidance:
        negative_prompt_embeds = torch.zeros_like(pose_embeddings)
        pose_embeddings = torch.cat([negative_prompt_embeds, pose_embeddings])

    return pose_embeddings.detach()

def _encode_image_with_pose(image_encoder, cc_projection, image, pose, device, dtype, do_classifier_free_guidance):
    img_prompt_embeds = _encode_image(image_encoder, image, device, dtype, False)
    pose_prompt_embeds = _encode_pose(pose, device, dtype, False)
    prompt_embeds = torch.cat([img_prompt_embeds, pose_prompt_embeds], dim=-1)
    prompt_embeds = cc_projection(prompt_embeds)
    # follow 0123, add negative prompt, after projection
    if do_classifier_free_guidance:
        negative_prompt = torch.zeros_like(prompt_embeds)
        prompt_embeds = torch.cat([negative_prompt, prompt_embeds])
    return prompt_embeds

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id


    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", revision=args.revision)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision)
    feature_extractor = None #CLIPFeatureExtractor.from_pretrained(args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision)

    vae.train()
    image_encoder.train()
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    # zero init unet conv_in from 4 channels to 8 channels
    conv_in_8 = torch.nn.Conv2d(8, unet.conv_in.out_channels, kernel_size=unet.conv_in.kernel_size, padding=unet.conv_in.padding)
    conv_in_8.requires_grad_(False)
    unet.conv_in.requires_grad_(False)
    torch.nn.init.zeros_(conv_in_8.weight)
    conv_in_8.weight[:,:4,:,:].copy_(unet.conv_in.weight)
    conv_in_8.bias.copy_(unet.conv_in.bias)
    unet.conv_in = conv_in_8
    unet.requires_grad_(True)
    unet.train()

    # zero init cc_projection
    cc_projection = CCProjection()
    torch.nn.init.eye_(list(cc_projection.parameters())[0][:768, :768])
    torch.nn.init.zeros_(list(cc_projection.parameters())[1])
    cc_projection.requires_grad_(True)
    cc_projection.train()

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            vae.enable_tiling()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"UNet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    if accelerator.unwrap_model(cc_projection).dtype != torch.float32:
        raise ValueError(
            f"UNet loaded as datatype {accelerator.unwrap_model(cc_projection).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    optimizer = optimizer_class(
        [{"params": unet.parameters(), "lr": args.learning_rate},
         {"params": cc_projection.parameters(), "lr": 10.*args.learning_rate}],
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon
    )
    # count total number of parameters in optimizer
    # unet 859532484
    # cc_projection 593664

    # print model info, learnable parameters, non-learnable parameters, total parameters, model size, all in billion
    def print_model_info(model):
        print("="*20)
        # print model class name
        print("model name: ", type(model).__name__)
        # print("model: ", model)
        print("learnable parameters(M): ", sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6)
        print("non-learnable parameters(M): ", sum(p.numel() for p in model.parameters() if not p.requires_grad) / 1e6)
        print("total parameters(M): ", sum(p.numel() for p in model.parameters()) / 1e6)
        print("model size(MB): ", sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024)



    print_model_info(unet)
    print_model_info(cc_projection)
    print_model_info(vae)
    print_model_info(image_encoder)

    # print total

    # Init Dataset
    image_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((args.resolution, args.resolution)),  # 256, 256
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
    )
    train_dataset = ObjaverseData(root_dir=args.train_data_dir, image_transforms=image_transforms, validation=False)
    validation_dataset = ObjaverseData(root_dir=args.train_data_dir, image_transforms=image_transforms, validation=True)
    # for training
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    # for validation set logs
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=1,
    )
    # for training set logs
    train_log_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=1,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # test with torch 2.0.0 by setting accelerator dynamo to induce, will result in NAN loss
    # # if torch version >=2.0 do compile
    # if version.parse(torch.__version__) >= version.parse("2.0.0"):
    #     print("Run torch compile")
    #     unet = torch.compile(unet, mode="reduce-overhead", fullgraph=True)

    # Prepare everything with our `accelerator`.
    unet, cc_projection, optimizer, train_dataloader, validation_dataloader, train_log_dataloader, lr_scheduler = accelerator.prepare(
        unet, cc_projection, optimizer, train_dataloader, validation_dataloader, train_log_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, image_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    do_classifier_free_guidance = args.guidance_scale > 1.0
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f" do_classifier_free_guidance = {do_classifier_free_guidance}")
    logger.info(f" conditioning_dropout_prob = {args.conditioning_dropout_prob}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        loss_epoch = 0.0
        num_train_elems = 0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet, cc_projection):
                # Convert images to latent space
                gt_image = batch["image_target"].to(dtype=weight_dtype)
                input_image = batch["image_cond"].to(dtype=weight_dtype)
                pose = batch["T"].to(dtype=weight_dtype)

                gt_latents = vae.encode(gt_image).latent_dist.sample().detach()
                gt_latents = gt_latents * vae.config.scaling_factor # follow zero123, only target image latent is scaled

                img_latents = vae.encode(input_image).latent_dist.mode().detach()   # .sample()

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(gt_latents)
                bsz = gt_latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=gt_latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(gt_latents.to(dtype=torch.float32), noise.to(dtype=torch.float32), timesteps).to(dtype=img_latents.dtype)

                if do_classifier_free_guidance:  #support classifier-free guidance, randomly drop out 5%
                    # Conditioning dropout to support classifier-free guidance during inference. For more details
                    # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                    random_p = torch.rand(bsz, device=gt_latents.device)
                    # Sample masks for the edit prompts.
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)

                    img_prompt_embeds = _encode_image(image_encoder, input_image, gt_latents.device, gt_latents.dtype, False)
                    pose_prompt_embeds = _encode_pose(pose, gt_latents.device, gt_latents.dtype, False)

                    # Final text conditioning.
                    null_conditioning = torch.zeros_like(img_prompt_embeds).detach()
                    img_prompt_embeds = torch.where(prompt_mask, null_conditioning, img_prompt_embeds)

                    prompt_embeds = torch.cat([img_prompt_embeds, pose_prompt_embeds], dim=-1)
                    prompt_embeds = cc_projection(prompt_embeds)

                    # Sample masks for the input images.
                    image_mask_dtype = img_latents.dtype
                    image_mask = 1 - (
                            (random_p >= args.conditioning_dropout_prob).to(image_mask_dtype)
                            * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                    )
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    # Final image conditioning.
                    img_latents = image_mask * img_latents
                else:
                    # Get the image_with_pose embedding for conditioning
                    prompt_embeds = _encode_image_with_pose(image_encoder, cc_projection, input_image, pose, gt_latents.device, weight_dtype, False)


                latent_model_input = torch.cat([noisy_latents, img_latents], dim=1)

                # Predict the noise residual
                model_pred = unet(
                    latent_model_input,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(gt_latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = (loss.mean([1, 2, 3])).mean()

                accelerator.backward(loss)
                # if accelerator.sync_gradients:
                #     params_to_clip = (itertools.chain(unet.parameters(), cc_projection.parameters()))
                #     accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if validation_dataloader is not None and global_step % args.validation_steps == 0:
                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
                        image_logs = log_validation(
                            validation_dataloader,
                            vae,
                            image_encoder,
                            feature_extractor,
                            unet,
                            cc_projection,
                            args,
                            accelerator,
                            weight_dtype,
                            'val',
                        )
                        if args.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_unet.restore(unet.parameters())
                    if train_log_dataloader is not None and (global_step % args.validation_steps == 0 or global_step == 1):
                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
                        train_image_logs = log_validation(
                            train_log_dataloader,
                            vae,
                            image_encoder,
                            feature_extractor,
                            unet,
                            cc_projection,
                            args,
                            accelerator,
                            weight_dtype,
                            'train',
                        )
                        if args.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_unet.restore(unet.parameters())
            loss_epoch += loss.detach().item()
            num_train_elems += 1
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "loss_epoch": loss_epoch/num_train_elems,
                    "epoch": epoch}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())
        # unet.save_pretrained(args.output_dir)
        # cc_projection = accelerator.unwrap_model(cc_projection)
        # cc_projection.save_pretrained(args.output_dir)

        pipeline = Zero1to3StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=accelerator.unwrap_model(vae),
            image_encoder=accelerator.unwrap_model(image_encoder),
            feature_extractor=feature_extractor,
            unet=unet,
            cc_projection=accelerator.unwrap_model(cc_projection),
            scheduler=noise_scheduler,
            safety_checker=None,
            torch_dtype=torch.float32,
        )
        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
