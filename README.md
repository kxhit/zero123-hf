# Zero-1-to-3: Zero-shot One Image to 3D Object

A HuggingFace diffusers implementation of [Zero123](https://github.com/cvlab-columbia/zero123)

##  Usage
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

Download original Zero123 checkpoint under `ckpts` through one of the following sources:

```
https://drive.google.com/drive/folders/1geG1IO15nWffJXsmQ_6VLih7ryNivzVs?usp=sharing
https://huggingface.co/cvlab/zero123-weights
wget https://cv.cs.columbia.edu/zero123/assets/$iteration.ckpt    # iteration = [105000, 165000, 230000, 300000]
```

Hugging Face diffusers weights are converted by script:
```commandline
python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path ./ckpt/165000.ckpt --dump_path ./zero123 --num_in_channels 8
```
Weights are hosted here:
```commandline
https://huggingface.co/kxic/zero123-105000
https://huggingface.co/kxic/zero123-165000
https://huggingface.co/kxic/zero123-xl
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
WIP, staytuned

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