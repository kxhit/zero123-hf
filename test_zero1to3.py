import os
import torch
from pipeline_zero1to3 import Zero1to3StableDiffusionPipeline
from diffusers.utils import load_image
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
import PIL

# download ckpt from "cvlab/zero123-weights/"
init_model_id = "lambdalabs/sd-image-variations-diffusers"
ckpt_dir = "./ckpts"
model_id = "kxic/zero123-105000" # zero123-105000, zero123-165000, zero123-xl
if 'xl' in model_id:
    ckpt = "zero123-xl"
else:
    ckpt = model_id.split('-')[-1]
ckpt_path = os.path.join(ckpt_dir, ckpt + ".ckpt")
assert model_id in ["kxic/zero123-105000", "kxic/zero123-165000", "kxic/zero123-xl"]
assert os.path.exists(ckpt_path)

image_encoder = CLIPVisionModelWithProjection.from_pretrained(init_model_id, subfolder="image_encoder", revision="v2.0")
feature_extractor = CLIPImageProcessor.from_pretrained(init_model_id, subfolder="feature_extractor", revision="v2.0")
vae = AutoencoderKL.from_pretrained(init_model_id, subfolder="vae", revision="v2.0")
scheduler = DDIMScheduler.from_pretrained(init_model_id, subfolder="scheduler", revision="v2.0")
# load 0123 unet weights, conv_in = 8, during training first 4 is inited from image variants ckpt, last 4 is inited from zero_init
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")

pipe = Zero1to3StableDiffusionPipeline.from_pretrained(init_model_id, torch_dtype=torch.float32, vae=vae,
                                                       image_encoder=image_encoder, feature_extractor=feature_extractor,
                                                       text_encoder=None, unet=unet, scheduler=scheduler)

# load cc_projection layer 772 (768+4) -> 768 todo convert model/weights to diffusers?
ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
cc_projection_weights = {"weight": ckpt["cc_projection.weight"], "bias": ckpt["cc_projection.bias"]}
pipe.load_cc_projection(cc_projection_weights)

pipe.enable_xformers_memory_efficient_attention()
pipe.enable_vae_tiling()
pipe.enable_attention_slicing()
pipe = pipe.to("cuda")
# todo hacky manually handle new module
pipe.cc_projection = pipe.cc_projection.to(pipe.device)

# set to eval mode
pipe.unet.eval()
pipe.vae.eval()
pipe.image_encoder.eval()
pipe.cc_projection.eval()

# test inference pipeline
# x y z, Polar angle (vertical rotation in degrees) 	Azimuth angle (horizontal rotation in degrees) 	Zoom (relative distance from center)
query_pose1 = [-75.0, 100.0, 0.0]
query_pose2 = [-20.0, 125.0, 0.0]
query_pose3 = [-55.0, 90.0, 0.0]

# load image
# H, W = (256, 256) # H, W = (512, 512)   # zero123 training is 256,256

# for batch input
input_image1 = load_image("./demo/4_blackarm.png") #load_image("https://cvlab-zero123-live.hf.space/file=/home/user/app/configs/4_blackarm.png")
input_image2 = load_image("./demo/8_motor.png") #load_image("https://cvlab-zero123-live.hf.space/file=/home/user/app/configs/8_motor.png")
input_image3 = load_image("./demo/7_london.png") #load_image("https://cvlab-zero123-live.hf.space/file=/home/user/app/configs/7_london.png")
input_images = [input_image1, input_image2, input_image3]
query_poses = [query_pose1, query_pose2, query_pose3]

# # for single input
# input_images = input_image2
# query_poses = query_pose2

# better do preprocessing
from gradio_new import preprocess_image, create_carvekit_interface
import numpy as np
import PIL.Image as Image
pre_images = []
models = dict()
print('Instantiating Carvekit HiInterface...')
models['carvekit'] = create_carvekit_interface()
if not isinstance(input_images, list):
    input_images = [input_images]
for raw_im in input_images:
    input_im = preprocess_image(models, raw_im, True)
    H, W = input_im.shape[:2]
    pre_images.append(Image.fromarray((input_im * 255.0).astype(np.uint8)))
input_images = pre_images

# infer pipeline, in original zero123 num_inference_steps=76
num_images_per_prompt = 4
images = pipe(input_imgs=input_images, prompt_imgs=input_images, poses=query_poses, height=H, width=W,
              guidance_scale=3.0, num_images_per_prompt=num_images_per_prompt, num_inference_steps=50).images

# save imgs
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
bs = len(input_images)
i = 0
for obj in range(bs):
    for idx in range(num_images_per_prompt):
        images[i].save(os.path.join(log_dir,f"obj{obj}_{idx}.jpg"))
        i += 1
