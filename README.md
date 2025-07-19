<div align="center">
	<img align="middle" src="imgs/crystalformer.png" width="400" alt="logo"/>
  <h2> Crystal Generation with Space Group Informed Transformer</h2> 
</div>

[![arXiv](https://img.shields.io/badge/arXiv-2403.15734-b31b1b.svg)](https://arxiv.org/abs/2403.15734) [![arXiv](https://img.shields.io/badge/arXiv-2504.02367-b31b1b.svg)](https://arxiv.org/abs/2504.02367)

_CrystalFormer_ is a transformer-based autoregressive model specifically designed for space group-controlled generation of crystalline materials. The space group symmetry significantly simplifies the
crystal space, which is crucial for data and compute efficient generative modeling of crystalline materials.

<div align="center">
  <img align="middle" src="imgs/output.gif" width="400">
  <h3> Generating Cs<sub>2</sub>ZnFe(CN)<sub>6</sub> Crystal (<a href=https://next-gen.materialsproject.org/materials/mp-570545>mp-570545</a>) </h3>
</div>

## Contents

- [Contents](#contents)
- [Model card](#model-card)
- [Status](#status)
- [Get Started](#get-started)
- [Installation](#installation)
  - [CPU installation](#cpu-installation)
  - [CUDA (GPU) installation](#cuda-gpu-installation)
  - [install required packages](#install-required-packages)
  - [command line tools](#command-line-tools)
- [Available Weights](#available-weights)
- [How to run](#how-to-run)
  - [train](#train)
  - [sample](#sample)
  - [evaluate](#evaluate)
- [Reinforcement Fine-tuning](#reinforcement-fine-tuning)
  - [$E\_{hull}$ Reward](#e_hull-reward)
  - [Dielectric FoM Reward](#dielectric-fom-reward)
- [How to cite](#how-to-cite)

## Model card

The model is an autoregressive transformer for the space group conditioned crystal probability distribution `P(C|g) = P (W_1 | ... ) P ( A_1 | ... ) P(X_1| ...) P(W_2|...) ... P(L| ...)`, where

- `g`: space group number 1-230
- `W`: Wyckoff letter ('a', 'b',...,'A')
- `A`: atom type ('H', 'He', ..., 'Og')
- `X`: factional coordinates
- `L`: lattice vector [a,b,c, alpha, beta, gamma]
- `P(W_i| ...)` and `P(A_i| ...)`  are categorical distributuions.
- `P(X_i| ...)` is the mixture of von Mises distribution.
- `P(L| ...)`  is the mixture of Gaussian distribution.

We only consider symmetry inequivalent atoms. The remaining atoms are restored based on the space group and Wyckoff letter information. Note that there is a natural alphabetical ordering for the Wyckoff letters, starting with 'a' for a position with the site-symmetry group of maximal order and ending with the highest letter for the general position. The sampling procedure starts from higher symmetry sites (with smaller multiplicities) and then goes on to lower symmetry ones (with larger multiplicities). Only for the cases where discrete Wyckoff letters can not fully determine the structure, one needs to further consider factional coordinates in the loss or sampling.

## Status

Major milestones are summarized below.
- v0.4.2 : Add implementation of direct preference optimization.
- v0.4.1 : Replace the absolute positional embedding with the Rotary Positional Embedding (RoPE).
- v0.4 : Add reinforcement learning (proximal policy optimization).
- v0.3 : Add conditional generation in the plug-and-play manner.
- v0.2 : Add Markov chain Monte Carlo (MCMC) sampling for template-based structure generation.
- v0.1 : Initial implementations of crystalline material generation conditioned on the space group.

## Get Started

**Notebooks**: The quickest way to get started with _CrystalFormer_ is our notebooks in the Google Colab and Bohrium (Chinese version) platforms:

- CrystalFormer Quickstart [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IMQV6OQgIGORE8FmSTmZuC5KgQwGCnDx?usp=sharing) [![Open In Bohrium](https://cdn.dp.tech/bohrium/web/static/images/open-in-bohrium.svg)](https://nb.bohrium.dp.tech/detail/68177247598): GUI notebook demonstrating the conditional generation of crystalline materials with _CrystalFormer_
- CrystalFormer Application [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QdkELaQXAHR1zEu2fcdfgabuoP61_wbU?usp=sharing): Generating stable crystals with a given structure prototype. This workflow can be applied to tasks that are dominated by element substitution
- CrystalFormer-RL [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ojSqMQzdnlWZRPOQP20nTvvIh67HXdwp#scrollTo=lKOZgUczOAxE) [![Open In Bohrium](https://cdn.dp.tech/bohrium/web/static/images/open-in-bohrium.svg)](https://bohrium.dp.tech/notebooks/52828216135): Reinforcement fine-tuning for materials design

## Installation

Create a new environment and install the required packages, we recommend using python `3.10.*` and conda to create the environment:

```bash
  conda create -n crystalgpt python=3.10
  conda activate crystalgpt
```

Before installing the required packages, you need to install `jax` and `jaxlib` first.

### CPU installation

```bash
pip install -U "jax[cpu]"
```

### CUDA (GPU) installation

If you intend to use CUDA (GPU) to speed up the training, it is important to install the appropriate version of `jax` and `jaxlib`. It is recommended to check the [jax docs](https://github.com/google/jax?tab=readme-ov-file#installation) for the installation guide. The basic installation command is given below:

```bash
pip install --upgrade pip

# NVIDIA CUDA 12 installation
# Note: wheels only available on linux.
pip install -U "jax[cuda12]"
```

### install required packages

```bash
pip install -r requirements.txt
```

### command line tools
To use the command line tools, you need to install the `crystalformer` package. You can use the following command to install the package:

```bash
pip install .
```

## Available Weights

We release the weights of the model trained on the MP-20 dataset and Alex-20 dataset. More details can be seen in the [model](./model/README.md) folder.

## How to run

### train

```bash
python ./main.py --folder ./data/ --train_path YOUR_PATH/mp_20/train.csv --valid_path YOUR_PATH/mp_20/val.csv
```

- `folder`: the folder to save the model and logs
- `train_path`: the path to the training dataset
- `valid_path`: the path to the validation dataset
- `test_path`: the path to the test dataset

### sample

```bash
python ./main.py --optimizer none --restore_path YOUR_MODEL_PATH --spacegroup 160 --num_samples 1000  --batchsize 1000 --temperature 1.0
```

- `optimizer`: the optimizer to use, `none` means no training, only sampling
- `restore_path`: the path to the model weights
- `spacegroup`: the space group number to sample
- `num_samples`: the number of samples to generate
- `batchsize`: the batch size for sampling
- `temperature`: the temperature for sampling

You can also use the `elements` to sample the specific element. For example, `--elements La Ni O` will sample the structure with La, Ni, and O atoms. The sampling results will be saved in the `output_LABEL.csv` file, where the `LABEL` is the space group number `g` specified in the command `--spacegroup`.

The input for the `elements` can be also the `json` file which specifies the atom mask in each Wyckoff site and the constraints. An example `atoms.json` file can be seen in the [data](./data/atoms.json) folder. There are two keys in the `atoms.json` file:

- `atom_mask`: set the atom list for each Wyckoff position, the element can only be selected from the list in the corresponding Wyckoff position
- `constraints`: set the constraints for the Wyckoff sites in the sampling, you can specify the pair of Wyckoff sites that should have the same elements


### evaluate

Before evaluating the generated structures, you need to transform the generated `g, W, A, X, L` to the `cif` format. You can use the following command to transform the generated structures to the `cif` format and save as the `csv` file:

```bash
python ./scripts/awl2struct.py --output_path YOUR_PATH --label SPACE_GROUP  --num_io_process 40
```

- `output_path`: the path to read the generated `L, W, A, X` and save the `cif` files
- `label`: the label to save the `cif` files, which is the space group number `g`
- `num_io_process`: the number of processes

> [!IMPORTANT]
> The following evaluation script requires the [`SMACT`](https://github.com/WMD-group/SMACT), [`matminer`](https://github.com/hackingmaterials/matminer), and [`matbench-genmetrics`](https://github.com/sparks-baird/matbench-genmetrics) packages. We recommend installing them in a separate environment to avoid conflicts with other packages.

Calculate the structure and composition validity of the generated structures:

```bash
python ./scripts/compute_metrics.py --root_path YOUR_PATH --filename YOUR_FILE --num_io_process 40
```

- `root_path`: the path to the dataset
- `filename`: the filename of the generated structures
- `num_io_process`: the number of processes

Calculate the novelty and uniqueness of the generated structures:

```bash
python ./scripts/compute_metrics_matbench.py --train_path TRAIN_PATH --test_path TEST_PATH --gen_path GEN_PATH --output_path OUTPUT_PATH --label SPACE_GROUP --num_io_process 40
```

- `train_path`: the path to the training dataset
- `test_path`: the path to the test dataset
- `gen_path`: the path to the generated dataset
- `output_path`: the path to save the metrics results
- `label`: the label to save the metrics results, which is the space group number `g`
- `num_io_process`: the number of processes

Note that the training, test, and generated datasets should contain the structures within the **same** space group `g` which is specified in the command `--label`.

More details about the post-processing can be seen in the [scripts](./scripts/README.md) folder.

## Reinforcement Fine-tuning

> [!IMPORTANT]
> Before running the reinforcement fine-tuning, please make sure you have installed the corresponding machine learning force field model or property prediction model. The `mlff_model` and `mlff_path` arguments in the command line should be set according to the model you are using. Now we support the[`orb`](https://github.com/orbital-materials/orb-models) and [`MACE`](https://github.com/ACEsuit/mace) models for the $E_{hull}$ reward, and the [`matgl`](https://github.com/materialsvirtuallab/matgl) model for the dielectric FoM reward.

### $E_{hull}$ Reward

```bash
train_ppo --folder ./data/\
          --restore_path YOUR_PATH\
          --valid_path YOUR_PATH/alex_20/val.csv\
          --test_path YOUR_PATH/alex_20/train.csv\
          --reward ehull\
          --convex_path YOUR_PATH/convex_hull_pbe_2023.12.29.json.bz2\
          --mlff_model orb\
          --mlff_path YOUR_PATH/orb-v2-20241011.ckpt
```

- `folder`: the folder to save the model and logs
- `restore_path`: the path to the pre-trained model weights
- `valid_path`: the path to the validation dataset
- `test_path`: the path to the test dataset. The space group distribution will be loaded from this dataset and used for the sampling in the reinforcement learning fine-tuning
- `reward`: the reward function to use, `ehull` means the energy above the convex hull
- `convex_path`: the path to the convex hull data, which is used to calculate the $E_{hull}$. Only used when the reward is `ehull`
- `mlff_model`: the machine learning force field model to predict the total energy. We support [`orb`](https://github.com/orbital-materials/orb-models) and [`MACE`](https://github.com/ACEsuit/mace) models for the $E_{hull}$ reward
- `mlff_path`: the path to load the checkpoint of the machine learning force field model

### Dielectric FoM Reward

```bash
train_ppo --folder ./data/\
          --restore_path YOUR_PATH\
          --valid_path YOUR_PATH/alex_20/val.csv\
          --test_path YOUR_PATH/alex_20/train.csv\
          --reward dielectric\
          --mlff_model matgl\
          --mlff_path YOUR_PATH/model1,YOUR_PATH/model2
```

- `folder`: the folder to save the model and logs
- `restore_path`: the path to the pre-trained model weights
- `valid_path`: the path to the validation dataset
- `test_path`: the path to the test dataset. The space group distribution will be loaded from this dataset and used for the sampling in the reinforcement learning fine-tuning
- `reward`: the reward function to use, `dielectric` means the dielectric figure of merit (FoM), which is the product of the total dielectric constant and the band gap
- `mlff_model`: the machine learning force field model to predict the total energy. We only support models in [`matgl`](https://github.com/materialsvirtuallab/matgl) for the dielectric reward
- `mlff_path`: the path to load the checkpoint of the machine learning force field model. Note that you need to provide the model paths for the total dielectric constant and band gap, separated by the `,`


## How to cite

```bibtex
@article{cao2024space,
      title={Space Group Informed Transformer for Crystalline Materials Generation}, 
      author={Zhendong Cao and Xiaoshan Luo and Jian Lv and Lei Wang},
      year={2024},
      eprint={2403.15734},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci}
}
```

```bibtex
@article{cao2025crystalformerrl,
      title={CrystalFormer-RL: Reinforcement Fine-Tuning for Materials Design}, 
      author={Zhendong Cao and Lei Wang},
      year={2025},
      eprint={2504.02367},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2504.02367}, 
}
```

**Note**: This project is unrelated to https://github.com/omron-sinicx/crystalformer with the same name.
