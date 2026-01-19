<a name="readme-top"></a>

<div align="center">
  <img src="./assets/LOGO.png" alt="Project Logo" width="300">
  <h1 align="center">Reasoning in the Dark: Interleaved Vision-Text Reasoning in Latent Space</h1>
</div>

<div align="center">

  <!-- Project Page -->
  <a href="https://github.com/ModalityDance/IVT-LR">
    <img src="https://img.shields.io/badge/Project-Page-6a5acd?style=for-the-badge" alt="Project Page">
  </a>

  <!-- Paper Link -->
  <a href="https://arxiv.org/abs/2510.12603">
    <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?style=for-the-badge&logo=arxiv" alt="Paper">
  </a>

  <!-- HuggingFace Models -->
  <a href="https://huggingface.co/FYYDCC/IVTLR">
    <img src="https://img.shields.io/badge/HuggingFace-Models-fcc21b?style=for-the-badge&logo=huggingface&logoColor=white" alt="HF Models">
  </a>

</div>


<!--
Overview

Points:

1. A short paragraph (2–4 sentences) describing:
    - What the project is.
    - The main purpose or capability.
    - What benefit users get.
    - The scope or application scenario.
    - The primary components included in this repository.

2. A "Key Features" section.  
   Each feature should include:
    - A short title (e.g., "Modular Design", "Fast Training").
    - A 1–2 sentence explanation of what the feature provides and why it matters.
   
3. Add more sections if needed.

4. A main figure image placed under assets/, e.g., assets/overview.png.  
   This image should visually summarize the system or framework.

-->

Interleaved Vision-Text Latent Reasoning (IVT-LR) is the first VLM framework that unifies textual and visual representations in the latent space and implements multimodal latent reasoning. Specifically, IVT-LR represents each reasoning step by combining two implicit parts: ***latent text*** and ***latent vision***. We further introduce a progressive multi-stage training strategy to enable MLLMs to perform the above multimodal latent reasoning steps.



<div align="center">
  <figure>
    <img src="./assets/image.png" alt="Overview" style="max-width: 100%; height: auto;">
    <br>
    <figcaption><em>Quick Overview of IVT-LR.</em></figcaption>
  </figure>
</div>


<!--
News 

Points:
1. Include chronological updates about the project.
2. Each news entry should have:
   - A date in [YYYY.MM] or [YYYY, MMM DD] format.
   - A short highlight sentence.
3. Optional but encouraged:
   - Bullet lists for detailed updates.
   - Links to papers, project pages, demos, datasets.
   - Emojis to increase readability.

-->

## 🔥 News 

<div style="max-height: 240px; overflow-y: auto;">

- **[2025.10]** Model files are now available on [Hugging Face](https://huggingface.co/FYYDCC/IVTLR) !

<!-- - **[2025.xx]** 🎉🎉 We released a major upgrade including new benchmarks, UI, and documentation.
  - 📄 Paper: <a href="{paper_link}">arXiv</a>
  - 📊 Benchmark Suite: <a href="{benchmark_link}">Link</a>
  - 🖥️ Web UI: {description} -->

- **[2025.10]** 🎉🎉Initial release of the project.

</div>


## 📑 Table of Contents <span id="table-of-contents"></span>


* <a href='#quick-start'>🚀 Quick Start</a>
  * <a href='#installation'>Installation</a>
  * <a href='#data'>Data Preparation</a>
  * <a href='#Training'>Training</a>
    * <a href='#Qwen2-VL'>Qwen2-VL</a>
    * <a href='#Chameleon'>Chameleon</a>
    * <a href='#Arguments'>Training Arguments</a>
  * <a href='#Inference'>Inference</a>
* <a href='#how-it-works'>✨ How It Works</a>
* <a href='#community'>🤝 Community</a>
* <a href='#acknowledgements'>🌱 Acknowledgements</a>
* <a href='#citation'>📚 Citation</a>


## 🚀 Quick Start <span id="quick-start"></span>


### 1. Installation <span id="installation"></span>

Clone repo:

```
git clone https://github.com/FYYDCC/IVT-LR.git
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
* {dataset_source_1}
* {dataset_source_2}



### 3. Training <span id="Training"></span>

#### Qwen2-VL on M3CoT <span id="Qwen2-VL"></span>

To train the Qwen2-VL model with IVT-LR on the M3CoT dataset:

```
cd qwen_vl
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_LEVEL=NVL   # if needed
PYTHONUNBUFFERED=1 nohup deepspeed --master_port 29501 qwenvl_run.py args/qwen.yaml --deepspeed --deepspeed_config ds_config.json > qwenvl.log 2>&1 &
```



#### Chameleon on ScienceQA <span id="Chameleon"></span>

For Chameleon:

```
cd chameleon
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_LEVEL=NVL   # if needed
PYTHONUNBUFFERED=1 nohup deepspeed --master_port 29501 chameleon_run_sqa.py args/chameleon.yaml --deepspeed --deepspeed_config ds_config.json > chameleon.log 2>&1 &
```


#### Training Arguments <span id="Arguments"></span>

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

### 4. Inference <span id="Inference"></span>

To generate the answer on the test split, run the inference code.

Qwen2-VL:

```
export CUDA_VISIBLE_DEVICES=0
nohup python infer.py > infer.log 2>&1 &    
```

Chameleon:

```
export CUDA_VISIBLE_DEVICES=0
nohup python infer_chameleon_scienceqa.py > infer.log 2>&1 &    
```

## ✨ How It Works <span id="how-it-works"></span>

🪐 **Project Name** is built around a modular research pipeline for **{core capability}**, where each component corresponds to a well-defined stage in the overall method.  
The system separates representation, reasoning, and output stages into independent modules, allowing controlled experimentation and analysis.  
This design enables flexible replacement of individual components without affecting the rest of the pipeline.

At a high level, the workflow proceeds as follows:

1. **{Step 1: Input processing}** — {Describe how raw inputs are converted into model-friendly representations.}  
2. **{Step 2: Core algorithm or modeling stage}** — {Explain how the main computation or retrieval happens.}  
3. **{Step 3: Final output generation}** — {Describe how results are composed, ranked, or produced.}

<div align="center">
  <figure>
    <img src="./assets/{method-figure.png}" alt="Method Overview" style="max-width: 100%; height: auto;">
    <br>
    <figcaption><em>Method overview of {Project Name}.</em></figcaption>
  </figure>
</div>


<!--
Community

REQUIRED:
1. Contributors section or GitHub contributors graph.
2. Star history chart.
3. A short paragraph encouraging engagement with the project.

OPTIONAL:
1. Social groups (Slack, Discord, WeChat, Feishu).
2. Issue tracker link (GitHub Issues).
3. Contribution guidelines (link to CONTRIBUTING.md if exists).

-->

## 🤝 Join the Community <span id="community"></span>

<div align="center">


<!-- Star history chart -->
[![Star History Chart](https://api.star-history.com/svg?repos=xxx/xxx&type=Date)](https://star-history.com/xxx/xxx&Date)

</div>


### 🔗 Related Projects

> **Note**: Please prioritize our own related papers. If additional projects are needed, refer to previous papers by group members to check whether they are directly relevant or comparable.


<div align="center">

<table>
<tr>
<td align="center">
  <b>🌟 Related Project 1</b><br/>
  <a href="{project_link_1}">{project_link_1}</a>
</td>
<td align="center">
  <b>🚀 Related Project 2</b><br/>
  <a href="{project_link_2}">{project_link_2}</a>
</td>
<td align="center">
  <b>🔧 Related Project 3</b><br/>
  <a href="{project_link_3}">{project_link_3}</a>
</td>
</tr>
</table>

</div>


## 📚 **Citation** <span id="citation"></span>

If you use **{Project Name}** in your research or applications, please consider citing:

```bibtex
@article{yourproject2025,
  title        = {{Project Name}: {Short descriptive subtitle}},
  author       = {Your Name and Collaborator Name and Others},
  journal      = {arXiv preprint arXiv:{xxxx.xxxxx}},
  year         = {2025}
}
```

<!-- Modify the repository URL accordingly. -->

<div align="center">

<a href="https://github.com/{github_org}/{repo_name}">
  <img src="https://img.shields.io/badge/⭐ Star%20us%20on%20GitHub-181717?style=for-the-badge&logo=github&logoColor=white" />
</a>

<a href="https://github.com/{github_org}/{repo_name}/issues">
  <img src="https://img.shields.io/badge/🐞 Report%20Issues-e74c3c?style=for-the-badge&logo=github" />
</a>

<a href="https://github.com/{github_org}/{repo_name}/discussions">
  <img src="https://img.shields.io/badge/💬 Discussions-20c997?style=for-the-badge&logo=github" />
</a>
<br/>
⭐ <b>Thank you for visiting {Project Name}!</b> ⭐

</div>
