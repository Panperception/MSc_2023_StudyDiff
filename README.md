The process of studying diffusion models involves transforming images from the training dataset into noise at different rates and assessing its impact on model training and image generation.

Need to configure a Linux system's root account with Python, Conda, CUDA, PyTorch, Jupyter server, and VSCode with Python and Jupyter plugins

todo tree default regex

```text
(//|#|<!--|;|/\*|^|^[ \t]*(-|\d+.))\s*($TAGS)
```

```text
C:\Users\{用户名}\AppData\Local\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe
```

three datasets

```python
r'/root/autodl-fs/dataset/huggan_smithsonian_butterflies_subset'
```

```python
r'/root/autodl-fs/dataset/nelorth-oxford-flowers'
```

```python
r'/root/autodl-fs/dataset/stanford_cars'
```

```python
r'/root/autodl-fs/dataset/cat_vs_dog'
```

# Setup Environment

hugging face token 

```text
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
```

open ai api key

```text
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
```


conda
```bash
conda init bash && source /root/.bashrc
```

conda create python env

```bash
source /etc/network_turbo
```

```bash
conda create -y -n pytorch_env python=3.8
```

```bash
conda activate pytorch_env
```

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

```bash
pip install \
 isort \
 black[jupyter] \
 jupyter \
 kaggle \
 transformers \
 datasets \
 einops \
 matplotlib \
 scipy \
 timm \
 open_clip_torch \
 lightning \
 wandb \
 torchmetrics[image] \
 opencv-contrib-python \
 xformers \
 controlnet_aux \
 diffusers \
 denoising_diffusion_pytorch
```

```bash
mkdir -p /root/.cache/torch/hub/checkpoints/
```

```bash
scp -P 17578 .\OneDrive\lancaster\data_science_msc_dissertation\weights-inception-2015-12-05-6726825d.pth root@connect.westb.seetacloud.com:/root/.cache/torch/hub/checkpoints/
```

# upload kaggle token

```bash
scp -P 17578 .\Downloads\kaggle.json root@connect.westb.seetacloud.com:/root/.kaggle/kaggle.json
```

```bash
kaggle datasets download -d dimensi0n/anime-faces-512
```

# stable diffusion web ui


```bash
cd autodl-tmp
```

```bash
rm -rf stable-diffusion-webui/
```

```bash
source /etc/network_turbo
```

```bash
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
```

```bash
cd stable-diffusion-webui/
```

```bash
bash webui.sh -f --port 6006
```

extension

- a1111-sd-webui-tagcomplete
- stable-diffusion-webui-localization-zh_Hans




# other

latex set path
```bash
# /root/.profile
export PATH=$PATH:/usr/local/texlive/2023/bin/x86_64-linux
# then restart system
```

proxy setting
```python
# https_proxy=https://172.20.0.113:12798
proxies={'https://': '172.20.0.113:12798'}
```

logger set

```bash
tail -f /root/autodl-tmp/app_logs/deep_learning.log
```

# run background

```bash
nohup python /root/lancaster_dissertation_code/main.py >/dev/null 2>&1 &
```

process check
reference https://man7.org/linux/man-pages/man1/ps.1.html
```bash
ps -ef
```
or
```bash
ps aux
```


format code
```bash
git config --global alias.fadd "! isort . && black --line-length 80 . && git add "
```




