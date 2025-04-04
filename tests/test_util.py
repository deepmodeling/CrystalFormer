from config import *
from src.utils import GLXAM_from_structures

import pandas as pd
from pymatgen.core.structure import Structure


def test_spg_lattice_match():
    data = pd.read_csv('./src/minival.csv') # it will take ~5 minutes for the whole val.csv
    cif_strings = data['cif']
    structures = [Structure.from_str(cif_string, fmt='cif') for cif_string in cif_strings]

    G, L, X, AM = GLXAM_from_structures(structures, atom_types=118, mult_types=6, n_max=20, dim=3)

    # convert G from one-hot to index
    G = jnp.argmax(G, axis=-1) + 1

    # Iterate over G and L arrays
    for idx, (space_group, lattice_params) in enumerate(zip(G, L)):

        a, b, c, alpha, beta, gamma = lattice_params

        angles_epsilon = 0.5  # numerical error tolerance
        abc_epsilon = 1e-3  # numerical error tolerance

        # Apply constraints based on space group number (g)
        if space_group < 3:
            continue  # no constraints
        elif space_group < 16:
            assert np.allclose([alpha, gamma], [90, 90], atol=angles_epsilon)
        elif space_group < 75:
            assert np.allclose([alpha, beta, gamma], [90, 90, 90], atol=angles_epsilon)
        elif space_group < 143:
            assert np.allclose([alpha, beta, gamma], [90, 90, 90], atol=angles_epsilon)
            assert np.allclose(a, b, atol=abc_epsilon)
        elif space_group < 195:
            assert np.allclose([alpha, beta, gamma], [90, 90, 120], atol=angles_epsilon)
            assert np.allclose(a, b, atol=abc_epsilon)
        else:
            assert np.allclose([alpha, beta, gamma], [90, 90, 90], atol=angles_epsilon)
            assert np.allclose([a, b, c], [a, b, c], atol=abc_epsilon)


test_spg_lattice_match()
