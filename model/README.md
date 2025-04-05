# Model Card

## Alex-20

The pre-trained model is available on [Google Drive](https://drive.google.com/file/d/1Fjt3bXzouAb-GX3ScAuejDA6eggtOYe4/view?usp=sharing) and [Hugging Face Model Hub](https://huggingface.co/zdcao/CrystalFormer/blob/main/alex20/PBE/epoch_005500.pkl). 

### Model Parameters

```python
params, transformer = make_transformer(
        key=jax.random.PRNGKey(42),
        Nf=5,
        Kx=16,
        Kl=4,
        n_max=21,
        h0_size=256,
        num_layers=16,
        num_heads=16,
        key_size=64,
        model_size=64,
        embed_size=32,
        atom_types=119,
        wyck_types=28,
        dropout_rate=0.1,
        attn_rate=0.1,
        widening_factor=4,
        sigmamin=1e-3
)
```

### Training dataset

Alex-20: contains ~1.3M general inorganic materials curated from the [Alexandria database](https://alexandria.icams.rub.de/), with $E_{hull} < 0.1$ eV/atom and no more than 20 atoms in unit cell. The dataset can be found in the [Google Drive](https://drive.google.com/drive/folders/1QeYz9lQX9Lk-OxhKBOwvuyKBecznPVlX?usp=drive_link) or [Hugging Face Datasets](https://huggingface.co/datasets/zdcao/alex-20).


## Alex-20 RL

- $E_{hull}$ reward: The checkpoint is available on [Google Drive](https://drive.google.com/file/d/1LlrpWj1GWUBZb-Ix_D3DfXxPd6EVsY6e/view?usp=sharing) and [Hugging Face Model Hub](https://huggingface.co/zdcao/CrystalFormer/blob/main/alex20/RL-ehull/epoch_000195.pkl). The reward is chosen to be the negative energy above the hull, which is calculated by the [Orb model](https://github.com/orbital-materials/orb-models) based on the Alexandria convex hull. 

- Dielectric FoM Reward: The checkpoint is available on [Google Drive](https://drive.google.com/file/d/1Jsa5uHa_Eu3cULqBDZxyia7CBgqe7Hg4/view?usp=sharing) and [Hugging Face Model Hub](https://huggingface.co/zdcao/CrystalFormer/blob/main/alex20/RL-dielectric/epoch_000100.pkl). The reward is chosen to be figures of dielectric figure of merit (FoM), which is the product of the total dielectric constant and the band gap. We use the pretrained [MEGNet](https://github.com/materialsvirtuallab/matgl/tree/main/pretrained_models/MEGNet-MP-2019.4.1-BandGap-mfi) to predict the band gap. The checkpoint of the total dielectric constant prediction model can be found in the [Google Drive](https://drive.google.com/drive/folders/1hQJD5R0dMJVC3nA1YkSHkCG9s-IAVNnA?usp=sharing). You can load the model using [matgl](https://github.com/materialsvirtuallab/matgl/tree/main) package. 


## MP-20

> [!IMPORTANT]   
> The load the MP-20 checkpoint, you need to switch the `CrystalFormer` to version 0.3 The current version of the model is not compatible with the MP-20 checkpoint.

### Checkpoint

The pre-trained model is available on [Google Drive](https://drive.google.com/file/d/1koHC6n38BqsY2_z3xHTi40HcFbVesUKd/view?usp=sharing) and [Hugging Face Model Hub](https://huggingface.co/zdcao/CrystalFormer/blob/main/mp20/epoch_003800.pkl).

### Model Parameters

```python
params, transformer = make_transformer(
        key=jax.random.PRNGKey(42),
        Nf=5,
        Kx=16,
        Kl=4,
        n_max=21,
        h0_size=256,
        num_layers=16,
        num_heads=16,
        key_size=64,
        model_size=64,
        embed_size=32,
        atom_types=119,
        wyck_types=28,
        dropout_rate=0.5,
        widening_factor=4,
        sigmamin=1e-3
)
```

### Training dataset

MP-20 (Jain et al., 2013): contains 45k general inorganic materials, including most experimentally known materials with no more than 20 atoms in unit cell.
More details can be found in the [CDVAE repository](https://github.com/txie-93/cdvae/tree/main/data/mp_20).