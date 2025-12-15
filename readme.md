# CNO2d with Transfer Learning

This repository is an extension of the original **Conditional Neural Operator (CNO)** implementation (NeurIPS 2023). We have integrated a **Transfer Learning** framework to efficiently adapt pre-trained CNO models to new downstream tasks.

## 🚀 Key Features: Transfer Learning Strategies

Beyond the original CNO training, this repository supports three transfer learning strategies to adapt a Source Model to a Target Model:

1.  **Fine-tuning (Full)**: Updates all parameters of the pre-trained model on the new dataset.
2.  **LoRA (Low-Rank Adaptation)**: Freezes the backbone and injects trainable rank-decomposition matrices, reducing the number of trainable parameters.
3.  **NLT (Non-Linear Transformation)**: A specialized transfer strategy constraining NLT weights and biases for efficient adaptation.

-----

## 🛠️ Prerequisites & Installation

> **Note:** This project inherits the strict environment requirements of the original CNO.

1.  **GPU Required:** Training is extremely slow on CPU. **Please run on a GPU.**
2.  **CUDA Toolkit:** Requires **CUDA Toolkit 11.1+** (system-level installation, not just Conda). [Download here](https://developer.nvidia.com/cuda-toolkit).
3.  **Compiler:**
      * Linux: GCC 7 or later.
      * Windows: Visual Studio compiler or PyCharm.

-----

## Workflow

The typical workflow involves two stages: **Pre-training** (Source Task) and **Transfer Learning** (Target Task).

### 1\. Pre-training (Source Task)

Train a standard CNO model on the source dataset using the original training script.

```bash
python TrainCNO.py
```

*Configure `which_example` in the script to select your source PDE (e.g., `navier_stokes`).*

### 2\. Transfer Learning (Target Task)

After obtaining a pre-trained model, choose one of the three strategies to transfer it to a target task.

#### Option A: Full Fine-tuning

To perform standard fine-tuning (updating all weights), run:

```bash
python Finetune_CNO.py
```

*This script loads the source model and resumes training on the target dataset with a lower learning rate.*

#### Option B: LoRA or NLT

To use parameter-efficient transfer learning strategies (LoRA or NLT), run the transfer script. You can specify the strategy within the script or via arguments (depending on your implementation).

```bash
# Example for LoRA or NLT strategy
python Transfer_CNO.py
```

  * **LoRA:** Inject low-rank adapters.
  * **NLT:** Constrains NLT weights ($\approx 1$) and biases ($\approx 0$) for stable transfer.

-----

## ⚙️ Configuration & Hyperparameters

### Model Parameters

Common hyperparameters for the CNO architecture:

| Parameter | Meaning |
| :--- | :--- |
| `N_layers` | Number of up/downsampling blocks |
| `channel_multiplier` | Regulates network width |
| `in_size` | Resolution of the computational grid |
| `activation` | `cno_lrelu` (recommended) or `lrelu` |

### Benchmark Datasets

Set the `which_example` variable in the scripts to select the PDE:

| `which_example` | PDE |
| :--- | :--- |
| `poisson` | Poisson equation |
| `wave_0_5` | Wave equation |
| `shear_layer` | Navier-Stokes equations |
| `darcy` | Darcy Flow |
| ... | (See `CNOBenchmarks.py` for full list) |

**Data Download:**
The datasets can be downloaded from [Zenodo](https://zenodo.org/records/10406879) (\~2.4GB).

-----

## Acknowledgments

This code is based on the original [CNO implementation](https://www.google.com/search?q=https://github.com/MachineLearningLifeScience/CNO) and uses filter implementations from [StyleGAN3](https://github.com/NVlabs/stylegan3).