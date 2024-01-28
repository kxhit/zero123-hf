import os
import torch
from pipeline_zero1to3 import Zero1to3StableDiffusionPipeline
from diffusers.utils import load_image

model_id = "kxic/zero123-165000" # zero123-105000, zero123-165000, zero123-xl, stable-zero123

pipe = Zero1to3StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
if "stable" in model_id:
    pipe.stable_zero123 = True  # stable-zero123 has difference in pose camera distance

pipe.enable_xformers_memory_efficient_attention()
pipe.enable_vae_tiling()
pipe.enable_attention_slicing()
pipe = pipe.to("cuda")

num_images_per_prompt = 4

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
# H, W = (256, 256)
# input_images = [input_image2.resize((H, W), PIL.Image.NEAREST)]
# query_poses = [query_pose2]


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
