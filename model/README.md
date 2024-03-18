## Pretrained model

The pre-trained model is available on [Google Drive](https://drive.google.com/file/d/1koHC6n38BqsY2_z3xHTi40HcFbVesUKd/view?usp=sharing). It can be downloaded using `wget`, `gdown`, or just by clicking the link. 


## Model Parameters

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

## Training dataset

MP-20 (Jain et al., 2013): contains 45k general inorganic materials, including most experimentally known materials with no more than 20 atoms in unit cell.
More details can be found in the [CDVAE repository](https://github.com/txie-93/cdvae/tree/main/data/mp_20).