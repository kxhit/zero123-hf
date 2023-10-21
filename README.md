# Zero-1-to-3: Zero-shot One Image to 3D Object

A HuggingFace Diffusers implementation of [Zero123](https://github.com/cvlab-columbia/zero123).

Merged into Diffusers Repo [here](https://github.com/huggingface/diffusers/blob/main/examples/community/pipeline_zero1to3.py).

##  Usage
Pytorch 2.0 for faster training and inference.
```
conda create -f environment.yml
```
or 
```
conda create -n zero123-hf python=3.9
conda activate zero123-hf
pip install -r requirements.txt
```

Install [xformer](https://github.com/facebookresearch/xformers#installing-xformers) properly to enable efficient transformers.
```commandline
conda install xformers -c xformers
# from source
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
```

Run diffusers pipeline demo:
```
python test_zero1to3.py
```

Run our gradio demo for novel view synthesis:

```
python gradio_new.py
```

##  Training
Download Zero123's Objaverse Renderings data:
```commandline
wget https://tri-ml-public.s3.amazonaws.com/datasets/views_release.tar.gz
```

Configure accelerator by
```commandline
accelerate config
```

Launch training:

Follow Original Zero123, fp32, gradient checkpointing, and EMA are turned on.
```commandline
accelerate launch train_zero1to3.py \
--train_data_dir /data/zero123/views_release \
--pretrained_model_name_or_path lambdalabs/sd-image-variations-diffusers \
--train_batch_size 192 \
--dataloader_num_workers 16 \
--output_dir logs \
--use_ema \
--gradient_checkpointing \
--mixed_precision no
```

While bf16/fp16 is also supported by running below
```commandline
accelerate launch train_zero1to3.py \
--train_data_dir /data/zero123/views_release \
--pretrained_model_name_or_path lambdalabs/sd-image-variations-diffusers \
--train_batch_size 192 \
--dataloader_num_workers 16 \
--output_dir logs \
--use_ema \
--gradient_checkpointing \
--mixed_precision bf16
```

For monitoring training progress, we recommand [wandb](https://wandb.ai/site) for its simplicity and powerful features.
```commandline
wandb login
```


##  Acknowledgement
This repository is based on original [Zero1to3](https://github.com/cvlab-columbia/zero123) and popular HuggingFace diffusion framework [diffusers](https://github.com/huggingface/diffusers).


##  Citation
If you find this work useful, a citation will be appreciated via:

```
@misc{zero123-hf,
    Author = {Xin Kong},
    Year = {2023},
    Note = {https://github.com/kxhit/zero123-hf},
    Title = {Zero123-hf: a diffusers implementation of zero123}
}

@misc{liu2023zero1to3,
      title={Zero-1-to-3: Zero-shot One Image to 3D Object}, 
      author={Ruoshi Liu and Rundi Wu and Basile Van Hoorick and Pavel Tokmakov and Sergey Zakharov and Carl Vondrick},
      year={2023},
      eprint={2303.11328},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
