---
language:
- en
license: mit
library_name: transformers
tags:
- vision-language
- navigation
- embodied-ai
- visual-navigation
- mixture-of-experts
- multimodal
- pytorch
datasets:
- R2R
- REVERIE
- RXR
- CVDN
- SOON
- ObjectNav-MP3D
metrics:
- success_rate
- spl
pipeline_tag: visual-question-answering
model-index:
- name: SAME
  results:
  - task:
      type: visual-navigation
      name: Vision-and-Language Navigation
    dataset:
      type: R2R
      name: Room-to-Room (R2R)
    metrics:
    - type: success_rate
      value: 76
      name: SR (val_unseen)
    - type: spl
      value: 66
      name: SPL (val_unseen)
    - type: success_rate
      value: 74
      name: SR (test_unseen)
    - type: spl
      value: 64
      name: SPL (test_unseen)
  - task:
      type: visual-navigation
      name: Vision-and-Language Navigation
    dataset:
      type: REVERIE
      name: REVERIE
    metrics:
    - type: success_rate
      value: 46.4
      name: SR (val_unseen)
    - type: spl
      value: 36.1
      name: SPL (val_unseen)
    - type: success_rate
      value: 48.6
      name: SR (test_unseen)
    - type: spl
      value: 37.1
      name: SPL (test_unseen)
  - task:
      type: visual-navigation
      name: Multilingual VLN
    dataset:
      type: RXR
      name: RxR-EN
    metrics:
    - type: success_rate
      value: 50.5
      name: SR (val_unseen)
    - type: ndtw
      value: 51.2
      name: nDTW (val_unseen)
  - task:
      type: visual-navigation
      name: Dialog Navigation
    dataset:
      type: CVDN
      name: CVDN
    metrics:
    - type: goal_progress
      value: 6.94
      name: GP (val)
    - type: goal_progress
      value: 7.07
      name: GP (test)
  - task:
      type: visual-navigation
      name: Object-Oriented Navigation
    dataset:
      type: SOON
      name: SOON
    metrics:
    - type: success_rate
      value: 36.1
      name: SR (val_unseen)
    - type: spl
      value: 25.4
      name: SPL (val_unseen)
    - type: success_rate
      value: 38.2
      name: SR (test_unseen)
    - type: spl
      value: 27.1
      name: SPL (test_unseen)
  - task:
      type: object-navigation
      name: Object Navigation
    dataset:
      type: ObjectNav-MP3D
      name: ObjectNav-MP3D
    metrics:
    - type: success_rate
      value: 76.3
      name: SR (val)
    - type: spl
      value: 42.7
      name: SPL (val)
---

# SAME: State-Adaptive Mixture of Experts for Language-Guided Visual Navigation

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv:2412.05552-red)](https://arxiv.org/abs/2412.05552)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-SAME-blue)](https://github.com/GengzeZhou/SAME)

</div>

## Model Description

**SAME** (State-Adaptive Mixture of Experts) is a unified framework for language-guided visual navigation that consolidates diverse navigation tasks into a single versatile agent. Unlike previous task-specific approaches, SAME can handle both **high-level category-specific search** (e.g., "find a chair") and **low-level language-guided navigation** (e.g., detailed turn-by-turn instructions) through a novel state-adaptive Mixture of Experts (MoE) architecture.

### Key Features

- **Multi-Task Capability**: Single model handles 9 different navigation datasets simultaneously
- **State-Adaptive MoE**: Dynamic expert routing based on multimodal features (text + visual observations)
- **Simulator-Free**: Works entirely with pre-computed CLIP ViT-B/16 features - no simulator installation required
- **Flexible Architecture**: MoE can be placed at attention query, key-value, or feed-forward network positions

## Model Architecture

SAME is built on a transformer-based architecture with the following key components:

| Component | Description |
|-----------|-------------|
| **Language Encoder** | 9-layer BERT-based transformer encoder |
| **Image Embeddings** | Processes 512-dim CLIP ViT-B/16 panoramic features |
| **Local VP Encoder** | Viewport-level information with crossmodal fusion |
| **Global Map Encoder** | Global spatial graph with dynamic routing |
| **State-Adaptive MoE** | 8 experts with top-2 selection, multimodal routing |

### MoE Routing

The State-Adaptive MoE uses multimodal features (fused text + visual embeddings) to dynamically route tokens to specialized experts. This allows the model to adapt its behavior based on:
- The granularity of language instructions
- Current visual observations
- Navigation task requirements

## Intended Uses

### Primary Use Cases

- **Vision-and-Language Navigation (VLN)**: Following natural language instructions in indoor environments
- **Object Navigation**: Finding target objects given category names
- **Dialog-based Navigation**: Multi-turn conversational navigation
- **Remote Object Grounding**: Navigating to and identifying remote objects

### Supported Tasks

| Task | Dataset | Description |
|------|---------|-------------|
| Low-Level Navigation | R2R, R2R-PREVALENT, R2R-ScaleVLN | Fine-grained instruction following |
| Object Grounding | REVERIE, REVERIE-ScaleVLN | Navigate and ground remote objects |
| Multilingual VLN | RXR-EN | Multilingual navigation (English) |
| Dialog Navigation | CVDN | Cooperative vision-and-dialog navigation |
| Object Search | SOON | Semantic object-oriented navigation |
| Object Navigation | ObjectNav-MP3D | Category-based object finding |

## How to Use

### Installation

```bash
git clone https://github.com/GengzeZhou/SAME.git
cd SAME
conda create --name SAME python=3.10
conda activate SAME
pip install -r requirements.txt
```

### Download Data and Models

```bash
# Download all datasets and features
python download.py --data

# Download pretrained models
python download.py --pretrain

# Download trained checkpoints (optional)
python download.py --checkpoints
```

### Training

```bash
cd src

# Single GPU training
python run.py --config_dir configs/main_multi_q.yaml

# Multi-GPU distributed training
torchrun --nproc_per_node=4 --master_port=29500 \
    run.py --config_dir configs/main_multi_q.yaml
```

### Evaluation

```bash
cd src
python run.py --config_dir configs/test.yaml \
    --options experiment.resume_file=/path/to/checkpoint.pt
```

### Configuration Options

```yaml
model:
  use_moe_layer: true
  moe_type: "Task"              # Task-based MoE
  moe_position: "Attn_q"        # Attn_q, Attn_kv, or FFN
  task_routing_feature: "multi" # Multimodal routing (recommended)
  num_experts: 8
  num_experts_per_tok: 2        # Top-2 expert selection
```

## Training Details

### Training Data

SAME is trained on 9 navigation datasets with weighted sampling:

| Dataset | Environment | Sampling Weight |
|---------|-------------|-----------------|
| R2R-ScaleVLN | HM3D | 10-20 |
| R2R-PREVALENT | MP3D | 1 |
| R2R | MP3D | 1 |
| REVERIE-ScaleVLN | HM3D | 1-10 |
| REVERIE | MP3D | 1 |
| RXR-EN | MP3D | 1 |
| CVDN | MP3D | 1 |
| SOON | MP3D | 1 |
| ObjectNav-MP3D | MP3D (Habitat) | 2 |

### Training Hyperparameters

- **Optimizer**: AdamW
- **Learning Rate**: 1e-5
- **Total Iterations**: 500,000
- **Batch Size**: 16
- **Gradient Clipping**: 0.5
- **Training Algorithm**: DAgger (Dataset Aggregation)
- **MoE Auxiliary Loss Coefficient**: 0.8

### Visual Features

- **Feature Extractor**: CLIP ViT-B/16
- **Feature Dimension**: 512
- **Format**: HDF5 / LMDB
- **Environments**: MatterSim, Habitat-MP3D, Habitat-HM3D

## Evaluation Results

SAME achieves state-of-the-art or highly competitive performance across all navigation benchmarks as a **unified model**, outperforming task-specific approaches in many cases.

### Main Results (Unified Model)

#### Room-to-Room (R2R)

| Split | SR ↑ | SPL ↑ |
|-------|------|-------|
| Val Unseen | **76** | 66 |
| Test Unseen | **74** | **64** |

#### REVERIE

| Split | SR ↑ | SPL ↑ |
|-------|------|-------|
| Val Unseen | **46.4** | **36.1** |
| Test Unseen | **48.6** | **37.1** |

#### RxR-EN (Multilingual VLN)

| Split | SR ↑ | nDTW ↑ |
|-------|------|--------|
| Val Unseen | **50.5** | **51.2** |

#### CVDN (Dialog Navigation)

| Split | GP ↑ |
|-------|------|
| Val | **6.94** |
| Test | 7.07 |

#### SOON (Object-Oriented Navigation)

| Split | SR ↑ | SPL ↑ |
|-------|------|-------|
| Val Unseen | 36.1 | 25.4 |
| Test Unseen | **38.2** | **27.1** |

#### ObjectNav-MP3D

| Split | SR ↑ | SPL ↑ |
|-------|------|-------|
| Val | **76.3** | 42.7 |

**Bold** indicates best performance among unified models. SAME achieves the best overall performance across all tasks with a single model.

### Evaluation Metrics

- **SR (Success Rate)**: Percentage of successful navigations (within 3m of goal)
- **SPL (Success weighted by Path Length)**: Efficiency-weighted success rate
- **nDTW (normalized Dynamic Time Warping)**: Path similarity to ground truth
- **GP (Goal Progress)**: Progress towards the goal in dialog navigation
- **NE (Navigation Error)**: Distance to goal at episode end
- **OSR (Oracle Success Rate)**: Success rate with oracle stop action

## Model Variants

| Variant | MoE Position | Routing | Checkpoint |
|---------|--------------|---------|------------|
| SAME-Q | Attention Query | Multimodal | `Attnq_pretrained_ckpt.pt` |
| SAME-KV | Attention K/V | Multimodal | `Attnkv_pretrained_ckpt.pt` |
| SAME-FFN | Feed-Forward | Multimodal | `FFN_pretrained_ckpt.pt` |

## Limitations

- **Indoor Environments Only**: Trained and evaluated on indoor navigation datasets
- **Pre-computed Features**: Requires pre-extracted CLIP features; cannot process raw images directly
- **English Language**: Primary support for English instructions (though RXR provides multilingual data)
- **Static Environments**: Assumes static environments without dynamic obstacles or agents

## Environmental Impact

- **Hardware**: Training conducted on NVIDIA A100 GPUs
- **Training Time**: Approximately 2-3 days on 4x A100 GPUs

## Citation

If you find this work helpful, please cite:

```bibtex
@article{zhou2024same,
  title={SAME: Learning Generic Language-Guided Visual Navigation with State-Adaptive Mixture of Experts},
  author={Gengze Zhou and Yicong Hong and Zun Wang and Chongyang Zhao and Mohit Bansal and Qi Wu},
  journal={arXiv preprint arXiv:2412.05552},
  year={2024},
}
```

## Authors

- **Gengze Zhou** - AIML, University of Adelaide ([Website](https://gengzezhou.github.io))
- **Yicong Hong** - Adobe Research ([Website](http://www.yiconghong.me))
- **Zun Wang** - UNC Chapel Hill ([Website](https://zunwang1.github.io))
- **Chongyang Zhao** - UNSW Sydney ([GitHub](https://github.com/zhaoc5))
- **Mohit Bansal** - UNC Chapel Hill ([Website](https://www.cs.unc.edu/~mbansal/))
- **Qi Wu** - University of Adelaide ([Website](http://www.qi-wu.me))

## Acknowledgements

We extend our gratitude to:
- [MatterPort3D](https://niessner.github.io/Matterport/) for the open-source platform
- [DUET](https://github.com/cshizhe/VLN-DUET) for the foundational architecture
- [ScaleVLN](https://github.com/wz0919/ScaleVLN) for augmented training data
- [NaviLLM](https://github.com/zd11024/NaviLLM) for additional insights

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue on the [GitHub repository](https://github.com/GengzeZhou/SAME) or contact the authors.
