## Hyperbolic Vision Transformers: Combining Improvements in Metric Learning

![image](https://github.com/solee0022/cv_hype/assets/126051717/1c266903-3534-4373-9a24-4ea8c2db7834)


### Code includes 
- [Proxy-Anchor](https://github.com/tjddus9597/Proxy-Anchor-CVPR2020) for datasets and evaluation (uses `pytorch_metric_learning`);
- [hyperbolic-image-embeddings](https://github.com/leymir/hyperbolic-image-embeddings) for hyperbolic operations;
- `train.py` - main training;
- `eval_pretrain.py` - encoder evaluation without training;
- `delta.py` - δ-hyperbolicity evaluation.


### Run training
```
python -m torch.distributed.launch --nproc_per_node=4 train.py  # multi GPU
python -m train --help  # single GPU
```

### Configs
```
python -m train --ds CUB --model vit_small_patch16_224 --num_samples 9 --lr 3e-5 --ep 50 --eval_ep "[50]" --resize 256
python -m train --ds CUB --model dino_vits16 --num_samples 9 --lr 1e-5 --ep 50 --eval_ep "[50]" --resize 256
python -m train --ds CUB --model deit_small_distilled_patch16_224 --num_samples 9 --lr 3e-5 --ep 50 --eval_ep "[50]" --resize 256

python -m train --ds Cars --model vit_small_patch16_224 --num_samples 9 --bs 882 --lr 3e-5 --ep 300 --eval_ep "[300]"
python -m train --ds Cars --model dino_vits16 --num_samples 9 --bs 882 --lr 1e-5 --ep 300 --eval_ep "[300]"
python -m train --ds Cars --model deit_small_distilled_patch16_224 --num_samples 9 --bs 882 --lr 3e-5 --ep 300 --eval_ep "[300]"

python -m train --ds SOP --model vit_small_patch16_224 --lr 3e-5 --ep 200 --eval_ep "[200]"
python -m train --ds SOP --model dino_vits16 --lr 1e-5 --ep 200 --eval_ep "[200]"
python -m train --ds SOP --model deit_small_distilled_patch16_224 --lr 3e-5 --ep 200 --eval_ep "[200]"

python -m train --ds Inshop --model vit_small_patch16_224 --lr 3e-5 --ep 400 --eval_ep "[400]"
python -m train --ds Inshop --model dino_vits16 --lr 1e-5 --ep 400 --eval_ep "[400]"
python -m train --ds Inshop --model deit_small_distilled_patch16_224 --lr 3e-5 --ep 400 --eval_ep "[400]"

# add --hyp_c 0 --t 0.1 for sphere version
# use --clip_r 0 to disable clipping
# use --eval_ep "r(300,410,10)" to evaluate every 10 epoch between 300 and 400
```

### Setup
```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

git clone https://github.com/NVIDIA/apex

pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex

pip install tqdm wandb timm typed-argument-parser pytorch_metric_learning

pip uninstall -y scipy && pip install scipy

wandb login
```

### Datasets
- [CUB-200](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)

