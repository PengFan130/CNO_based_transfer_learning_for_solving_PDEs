# CNO-based Transfer Learning for Solving PDEs

This is our code for paper "Convolutional-neural-operator-based transfer learning for solving PDEs". This repository is an extension of the original **Conditional Neural Operator (CNO)** implementation (NeurIPS 2023). We have integrated a **Transfer Learning** framework to efficiently adapt pre-trained CNO models to new downstream tasks.

## 🚀 Key Features: Transfer Learning Strategies

Beyond the original CNO training, this repository supports three transfer learning strategies to adapt a Source Model to a Target Model:

1.  **Fine-tuning **: Updates the last few layers of the pre-trained model on the new dataset.
2.  **LoRA (Low-Rank Adaptation)**: Freezes the backbone and injects trainable rank-decomposition matrices, reducing the number of trainable parameters.
3.  **NLT (Neuron Linear Transformation)**: A specialized transfer strategy performing a linear transformation on the parameters of the source model for efficient adaptation.

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

To perform fine-tuning, run:

```bash
python Finetune_CNO.py
```

*This script loads the source model and resumes training on the target dataset. This script also provides the flexibility to selectively fine-tune specific layers*

#### Option B: LoRA or NLT

To select the strategy, adjust the imports in `CNOBenchmarks.py`:

- For **NLT**: Use `from CNOModule_nlt import CNO as cno_nlt` (Line 15).
- For **LoRA**: Use `from CNOModule_lora import CNO as cno_nlt` (Line 17).

Then, use parameter-efficient transfer learning strategies (LoRA or NLT), run the transfer script. 

```bash
# Example for LoRA or NLT strategy
python Transfer_CNO.py
```

  * **LoRA:** Inject low-rank adapters.
  * **NLT:** Neuron linear transformation.

-----

## ⚙️ Configuration & Hyperparameters

### Model Parameters

Common hyperparameters for the CNO architecture:

| Parameter | Meaning |
| :--- | :--- |
| `N_layers` | Number of up/downsampling blocks |
| `channel_multiplier` | Regulates network width |
| `in_size` | Resolution of the computational grid |
| `activation` | `cno_torch_lrelu` (recommended) or `lrelu` |

### Benchmark Datasets

Set the `which_example` variable in the scripts to select the PDE:

| `which_example` | PDE |
| :--- | :--- |
| `apens`         | Navier-Stokes equations                |
| `ks` | Kuramoto-Sivashinsky equation |
| `burss`         | Brusselator diffusion-reaction system  |
| ...             | (See `CNOBenchmarks.py` for full list) |

**Data Download:**
The CNO datasets can be downloaded from [Zenodo](https://zenodo.org/records/10406879) (\~2.4GB).

The datasets used in this paper are **available for download at** [https://drive.google.com/file/d/1lRxDi9tzx_DqlMpvhsTvy8jNGsqspZ4U/view?usp=sharing] (~1.4GB).

-----

## Acknowledgments

This code is based on the original [CNO implementation](https://www.google.com/search?q=https://github.com/MachineLearningLifeScience/CNO) and uses NLT implementations from [NLT](https://github.com/taohan10200/NLT).

## Citation

If you use our models, code, or datasets, please consider citing our paper:

```bash
@misc{fan2025cnobasedtransferlearning,
      title={Convolutional-neural-operator-based transfer learning for solving PDEs}, 
      author={Peng Fan and Guofei Pang},
      year={2025},
      eprint={2512.17969},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.17969}, 
}
```