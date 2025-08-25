# VINCIE: Unlocking In-context Image Editing from Video
<p align="center">
  <a href="https://vincie2025.github.io/">
    <img
      src="https://img.shields.io/badge/VINCIE-Website-0A66C2?logo=safari&logoColor=white"
      alt="VINCIE Website"
    />
  </a>
  <a href="https://arxiv.org/abs/2506.10941">
    <img
      src="https://img.shields.io/badge/VINCIE-Paper-red?logo=arxiv&logoColor=red"
      alt="VINCIE Paper on ArXiv"
    />
  </a>
  <a href="https://github.com/ByteDance-Seed/VINCIE">
            <img 
              alt="Github" src="https://img.shields.io/badge/VINCIE-Codebase-536af5?color=536af5&logo=github"
              alt="VINCIE Codebase"
            />
  </a>
  <a href="https://huggingface.co/collections/ByteDance-Seed/vincie-6864cc2e3116d82e4a83a17c">
    <img 
        src="https://img.shields.io/badge/VINCIE-Models-yellow?logo=huggingface&logoColor=yellow" 
        alt="VINCIE Models"
    />
  </a>
  <a href="https://huggingface.co/datasets/leigangqu/VINCIE-10M">
    <img 
        src="https://img.shields.io/badge/VINCIE-Dataset-yellow?logo=huggingface&logoColor=yellow" 
        alt="VINCIE-10M Dataset"
    />
  </a>
   <!-- <a href="https://huggingface.co/spaces/ByteDance-Seed/VINCIE-3B">
    <img 
        src="https://img.shields.io/badge/VINCIE-Space-orange?logo=huggingface&logoColor=yellow" 
        alt="VINCIE Space"
    />
  </a> -->
</p>

> [Leigang Qu](https://leigang-qu.github.io/), [Feng Cheng](https://klauscc.github.io/), [Ziyan Yang](https://ziyanyang.github.io/), [Qi Zhao](https://kevinz8866.github.io/), [Shanchuan Lin](https://scholar.google.com/citations?user=EDWUw7gAAAAJ&hl=en), [Yichun Shi](https://seasonsh.github.io/), [Yicong Li](https://yl3800.github.io/), [Wenjie Wang](https://wenjiewwj.github.io/), [Tat-Seng Chua](https://www.chuatatseng.com/), [Lu Jiang](http://www.lujiang.info/index.html)
> 
> In-context image editing aims to modify images based on a contextual sequence comprising text and previously generated images. Existing methods typically depend on task-specific pipelines and expert models (*e.g.*, segmentation and inpainting) to curate training data. In this work, we explore whether an in-context image editing model can be learned directly from videos. We introduce a scalable approach to annotate videos as interleaved multimodal sequences. To effectively learn from this data, we design a block-causal diffusion transformer trained on three proxy tasks: next-image prediction, current segmentation prediction, and next-segmentation prediction. Additionally, we propose a novel multi-turn image editing benchmark to advance research in this area. Extensive experiments demonstrate that our model exhibits strong in-context image editing capabilities and achieves state-of-the-art results on two multi-turn image editing benchmarks. Despite being trained exclusively on videos, our model also shows promising abilities in multi-concept composition, story generation, and chain-of-editing applications.

<p align="center"><img src="assets/teaser.jpeg" width="95%"></p>


## News

- **25 Aug, 2025:** Released the official [website](https://vincie2025.github.io/) and the inference code.
- **23 Aug, 2025:** Released the [VINCIE-10M dataset](https://huggingface.co/datasets/leigangqu/VINCIE-10M). 
- **12 Jun, 2025:** Released the [VINCIE technical report](https://arxiv.org/abs/2506.10941) . 


## Quick Start

1️⃣  Set up environment
```bash
git clone https://github.com/ByteDance-Seed/VINCIE
cd VINCIE
conda create -n vincie python=3.10 -y
conda activate vincie
pip install -r requirements.txt
pip install flash_attn==2.6.3 --no-build-isolation
```

2️⃣  Download pretrained checkpoint
```python
from huggingface_hub import snapshot_download

save_dir = "ckpt/VINCIE-3B"
repo_id = "ByteDance-Seed/VINCIE-3B"
cache_dir = save_dir + "/cache"

snapshot_download(cache_dir=cache_dir,
  local_dir=save_dir,
  repo_id=repo_id,
  local_dir_use_symlinks=False,
  resume_download=True
)

```


## Inference
```bash
turn1="Change the color of the puppy to white, but keep the pink clothes and the hat unchanged."
turn2="Add a small black kitten sitting beside the puppy in the lower right corner."
turn3="Change the background to an indoor setting."
turn4="Make the dog open its mouth and bark loudly."
turn5="Change the image into a watercolor painting."
input_img="assets/dog.png"
output_dir="output/dog"

python main.py configs/generate.yaml \
    generation.positive_prompt.image_path=$input_img \
    generation.positive_prompt.prompts="[\"$turn1\", \"$turn2\", \"$turn3\", \"$turn4\", \"$turn5\"]" \
    generation.output.dir=$output_dir
```


## Citation

```bibtex
@article{qu2025vincie,
  title   = {VINCIE: Unlocking In-context Image Editing from Video},
  author  = {Qu, Leigang and Cheng, Feng and Yang, Ziyan and Zhao, Qi and Lin, Shanchuan and Shi, Yichun and Li, Yicong and Wang, Wenjie and Chua, Tat-Seng and Jiang, Lu},
  journal = {arXiv preprint arXiv:2506.10941},
  year    = {2025}
}
```

## License
This project is licensed under the [Apache-2.0 License](LICENSE), subject to any intellectual property rights in the model owned by ByteDance. The text encoder of the model is adapted from [Qwen-14B](https://huggingface.co/Qwen/Qwen-14B) and your use of that model must comply with its license. 