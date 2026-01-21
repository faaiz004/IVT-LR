<a name="readme-top"></a>

<div align="center">
  <h1 align="center">Reasoning in the Dark: Interleaved Vision-Text Reasoning in Latent Space</h1>
</div>

<div align="center">

<!-- Paper Link -->

<a href="https://arxiv.org/abs/2510.12603">
    <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?style=for-the-badge&logo=arxiv" alt="Paper">
  </a>

<!-- HuggingFace Models -->

<a href="https://huggingface.co/collections/ModalityDance/ivt-lr">
    <img src="https://img.shields.io/badge/HuggingFace-Models-fcc21b?style=for-the-badge&logo=huggingface&logoColor=white" alt="HF Models">
  </a>

<a href="https://huggingface.co/papers/2510.12603">
    <img src="https://img.shields.io/badge/HuggingFace-Papers-fcc21b?style=for-the-badge&logo=huggingface&logoColor=white" alt="HF Papers">
  </a>

</div>


Interleaved Vision-Text Latent Reasoning (IVT-LR) is the first VLM framework that unifies textual and visual representations in the latent space and implements multimodal latent reasoning. Specifically, IVT-LR represents each reasoning step by combining two implicit parts: ***latent text*** and ***latent vision***. We further introduce a progressive multi-stage training strategy to enable MLLMs to perform the above multimodal latent reasoning steps.

<div align="center">
  <figure>
    <img src="./assets/image.png" alt="Overview" style="max-width: 100%; height: auto;">
    <br>
    <figcaption><em>Quick Overview of IVT-LR.</em></figcaption>
  </figure>
</div>


## 🔥 News

<div style="max-height: 240px; overflow-y: auto;">

- **[2026.01]** Model files are now available on [Hugging Face](https://huggingface.co/collections/ModalityDance/ivt-lr) !

- **[2025.10]** 🎉🎉Initial release of the project.

</div>


## 📑 Table of Contents <span id="table-of-contents"></span>

* [🚀 Quick Start](#quick-start)
  * [Installation](#installation)
  * [Data Preparation](#data)
  * [Training](#training)
    * [Qwen2-VL](#qwen2-vl)
    * [Chameleon](#chameleon)
    * [Training Arguments](#arguments)
  * [Inference](#inference)
* [✨ How It Works](#how-it-works)
* [🔗 Related Projects](#related)
* [📚 Citation](#citation)


## 🚀 Quick Start <span id="quick-start"></span>

### 1. Installation <span id="installation"></span>

Clone repo:

```
git clone https://github.com/ModalityDance/IVT-LR.git
cd IVT-LR
```

Setup environment:

```
conda env create -f environment.yml
conda activate ivtlr
```

Expected folder structure

```plaintext
IVT-LR/
  ├── chameleon
        ├── args/
        ├── chameleon_dataset.py
        ├── ...
  ├── qwen_vl
        ├── args/
        ├── dataset.py
        ├── ...
  └── environment.yml
```

### 2. Data Preparation <span id="data"></span>

Download datasets:

```
dataset = load_dataset("LightChen2333/M3CoT")
dataset = load_dataset("derek-thomas/ScienceQA")
```

or download manually from:

* [M3CoT](https://huggingface.co/datasets/LightChen2333/M3CoT)
* [ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA)

### 3. Training <span id="training"></span>

> **💡 Skip Training:** If you want to skip training and directly run inference, you can download our pretrained models from the [IVT-LR Collection](https://huggingface.co/collections/ModalityDance/ivt-lr) on Hugging Face.

#### Qwen2-VL <span id="qwen2-vl"></span>

To train the Qwen2-VL model with IVT-LR on the M3CoT dataset:

```
cd qwen_vl
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_LEVEL=NVL   # if needed
PYTHONUNBUFFERED=1 nohup deepspeed --master_port 29501 qwenvl_run.py args/qwen.yaml --deepspeed --deepspeed_config ds_config.json > qwenvl.log 2>&1 &
```

To train the Qwen2-VL model with IVT-LR on the ScienceQA dataset:

```
cd qwen_vl
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_LEVEL=NVL   # if needed
PYTHONUNBUFFERED=1 nohup deepspeed --master_port 29501 qwenvl_run_sqa.py args/qwen.yaml --deepspeed --deepspeed_config ds_config.json > qwenvl.log 2>&1 &
```

#### Chameleon <span id="chameleon"></span>

For Chameleon on M3CoT:

```
cd chameleon
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_LEVEL=NVL   # if needed
PYTHONUNBUFFERED=1 nohup deepspeed --master_port 29501 chameleon_run.py args/chameleon.yaml --deepspeed --deepspeed_config ds_config.json > chameleon.log 2>&1 &
```

For Chameleon on ScienceQA:

```
cd chameleon
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_LEVEL=NVL   # if needed
PYTHONUNBUFFERED=1 nohup deepspeed --master_port 29501 chameleon_run_sqa.py args/chameleon.yaml --deepspeed --deepspeed_config ds_config.json > chameleon.log 2>&1 &
```

#### Training Arguments <span id="arguments"></span>

Key parameters in configuration:

- `save_path`: Checkpoint save directory
- `name`: Experiment name
- `epochs_per_stage`: Epochs per latent reasoning stage (default: 4)
- `max_latent_stage`: Maximum latent reasoning stages (default: 5)
- `resume`: Resume epoch number (default: 0)
- `batch_size_training`: Batch size per GPU (default: 4)
- `gradient_accumulation_steps`: Gradient accumulation steps (default: 4)
- `num_epochs`: Total training epochs (default: 16)
- `lr`: Learning rate (default: 4e-5)

### 4. Inference <span id="inference"></span>

To generate the answer on the test split, run the inference code.

Qwen2-VL on M3CoT:

```
export CUDA_VISIBLE_DEVICES=0
nohup python infer.py > infer.log 2>&1 &  
```

Qwen2-VL on ScienceQA:
```
export CUDA_VISIBLE_DEVICES=0
nohup python infer_sqa.py > infer.log 2>&1 &  
```

Chameleon on M3CoT:
```
export CUDA_VISIBLE_DEVICES=0
nohup python infer_chameleon.py > infer.log 2>&1 &  
```

Chameleon on ScienceQA:
```
export CUDA_VISIBLE_DEVICES=0
nohup python infer_chameleon_scienceqa.py > infer.log 2>&1 &  
```

## ✨ How It Works <span id="how-it-works"></span>

**IVT-LR** introduces a novel paradigm of multimodal latent reasoning that unifies textual and visual representations within the latent space. Unlike explicit chain-of-thought methods that require labor-intensive vision-text annotations, IVT-LR performs reasoning implicitly, achieving both annotation efficiency and inference speedup.

At a high level, the workflow proceeds as follows:

1. **Interleaved Multimodal Representation** — Each reasoning step combines two implicit components: ***latent text*** and ***latent vision***. This interleaved structure enables the model to jointly leverage both modalities during reasoning.

2. **Progressive Multi-Stage Training** — We employ a curriculum-style training strategy that gradually increases the number of latent reasoning stages. This progressive approach helps MLLMs learn to perform multimodal latent reasoning in a stable and effective manner.

3. **Dynamic Attention Allocation** — A key insight from our analysis is that interleaved multimodal reasoning leads to dynamic attention redistribution. As reasoning progresses, the model adaptively shifts attention between visual and textual tokens based on task demands, significantly enhancing visual perception capabilities.


## 🔗 **Related Projects** <span id="related"></span>

### 📄 Related Papers

- **[Coconut: Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769)**  
  A pioneering work on latent reasoning that uses continuous thought representations for LLM reasoning.

### 🌟 Awesome Collections

- **[Awesome Latent Space](https://github.com/YU-deep/Awesome-Latent-Space)**  
  A curated collection of resources on latent space methods and applications.

- **[Awesome Latent CoT](https://github.com/EIT-NLP/Awesome-Latent-CoT)**  
  A comprehensive list of latent chain-of-thought reasoning resources.


## 📚 **Citation** <span id="citation"></span>

If you use **IVT-LR** in your research or applications, please consider citing:

```bibtex
@article{chen2025reasoning,
  title={Reasoning in the dark: Interleaved vision-text reasoning in latent space},
  author={Chen, Chao and Ma, Zhixin and Li, Yongqi and Hu, Yupeng and Wei, Yinwei and Li, Wenjie and Nie, Liqiang},
  journal={arXiv preprint arXiv:2510.12603},
  year={2025}
}
```


<div align="center">

<a href="https://github.com/ModalityDance/IVT-LR">
  <img src="https://img.shields.io/badge/⭐ Star%20us%20on%20GitHub-181717?style=for-the-badge&logo=github&logoColor=white" />
</a>

<a href="https://github.com/ModalityDance/IVT-LR/issues">
  <img src="https://img.shields.io/badge/🐞 Report%20Issues-e74c3c?style=for-the-badge&logo=github" />
</a>

<br/>
⭐ <b>Thank you for visiting IVT-LR!</b> ⭐

</div>
