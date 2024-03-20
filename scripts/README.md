## Post-Processing Scripts

### Contents
- [Post-Processing Scripts](#post-processing-scripts)
  - [Contents](#contents)
  - [Transform](#transform)
  - [Structure and Composition Validity](#structure-and-composition-validity)
  - [Novelty and Uniqueness](#novelty-and-uniqueness)
  - [Relaxation](#relaxation)
  - [Energy Above the Hull](#energy-above-the-hull)
  - [Embedding Visualization](#embedding-visualization)
  - [Structure Visualization](#structure-visualization)

### Transform
`awl2struct.py` is a script to transform the generated `L, W, A, X` to the `cif` format. 

```bash
python awl2struct.py --output_path YOUR_PATH --label SPACE_GROUP --num_io_process 40
```
- `output_path`: the path to read the generated `L, W, A, X` and save the `cif` files
- `label`: the label to save the `cif` files, which is the space group number
- `num_io_process`: the number of processes


### Structure and Composition Validity
`compute_metrics.py` is a script to calculate the structure and composition validity of the generated structures.

```bash
python ../scripts/compute_metrics.py --root_path YOUR_PATH --filename YOUR_FILE --output_path ./ --num_io_process 40
```
- `root_path`: the path to the dataset
- `filename`: the filename of the generated structures
- `num_io_process`: the number of processes

### Novelty and Uniqueness
`compute_metrics_matbench.py` is a script to calculate the novelty and uniqueness of the generated structures.
```bash
python ../scripts/compute_metrics_matbench.py --train_path TRAIN_PATH --test_path TEST_PATH --gen_path GEN_PATH --output_path OUTPUT_PATH --label SPACE_GROUP --num_io_process 40
```
- `train_path`: the path to the training dataset
- `test_path`: the path to the test dataset
- `gen_path`: the path to the generated dataset
- `output_path`: the path to save the metrics results
- `label`: the label to save the metrics results, which is the space group number `g`
- `num_io_process`: the number of processes

Note that the training, test, and generated datasets should contain the structures within the **same** space group `g` which is specified in the command `--label`.


### Relaxation
`matgl_relax.py` is a script to relax the generated structures using the `matgl` package. You can install the `matgl` following the instructions in the [matgl repository](https://github.com/materialsvirtuallab/matgl?tab=readme-ov-file).
```bash
python matgl_relax.py --restore_path RESTORE_PATH --filename FILENAME --relaxation --model_path MODEL_PATH
```
- `restore_path`: the path to the generated structures
- `filename`: the filename of the generated structures
- `relaxation`: whether to relax the structures, if not specified, the script will only predict the energy of the structures without relaxation
- `model_path`: the path to the `matgl` model checkpoint


### Energy Above the Hull
`e_above_hull.py` is a script to calculate the energy above the hull of the generated structures based on the Materials Project database. To calculate the energy above the hull, the API key of the Materials Project is required, which can be obtained from the [Materials Project website](https://next-gen.materialsproject.org/). Furthermore, the `mp_api` package should be installed.

```bash
python e_above_hull.py --restore_path RESTORE_PATH --filename FILENAME --api_key API_KEY --label LABEL --relaxation
```
- `restore_path`: the path to the structures 
- `filename`: the filename of the structures
- `api_key`: the API key of the Materials Project
- `label`: the label to save the energy above the hull file
- `relaxation`: whether to calculate the energy above the hull based on the relaxed structures

### Embedding Visualization
`plot_embeddings.py` is a script to visualize the correlation of the learned embedding vectors of different elements.
    
```bash
python plot_embeddings.py --restore_path RESTORE_PATH
```

- `restore_path`: the path to the model checkpoint

### Structure Visualization
`structure_visualization.ipynb` is a notebook to visualize the generated structures.
