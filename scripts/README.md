# Convert original weights to diffusers

Download original Zero123 checkpoint under `ckpts` through one of the following sources:

```
https://drive.google.com/drive/folders/1geG1IO15nWffJXsmQ_6VLih7ryNivzVs?usp=sharing
https://huggingface.co/cvlab/zero123-weights
wget https://cv.cs.columbia.edu/zero123/assets/$iteration.ckpt    # iteration = [105000, 165000, 230000, 300000]
```

For Stable-Zero123, download from:
```
https://huggingface.co/stabilityai/stable-zero123
```

Hugging Face diffusers weights are converted by script:
```commandline
cd scripts
python convert_zero123_to_diffusers.py --checkpoint_path /path/zero123/105000.ckpt --dump_path ./zero1to3 --original_config_file /path/zero123/configs/sd-objaverse-finetune-c_concat-256.yaml
```

Weights are hosted here:
```commandline
# zero1to3
https://huggingface.co/kxic/zero123-105000
https://huggingface.co/kxic/zero123-165000
https://huggingface.co/kxic/zero123-xl

# Stable-Zero123
https://huggingface.co/kxic/stable-zero123  
```