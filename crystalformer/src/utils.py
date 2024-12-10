import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from pyxtal import pyxtal
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from functools import partial
import multiprocessing
import os

from ase.io import read
from spglib import get_layergroup
from pyxtal.lattice import Lattice as xLattice

from crystalformer.src.elements import element_list
from crystalformer.src.sym_group import *

@jax.vmap
def sort_atoms(W, A, X):
    """
    lex sort atoms according W, X, Y, Z

    W: (n, )
    A: (n, )
    X: (n, dim) int
    """
    W_temp = jnp.where(W>0, W, 9999) # change 0 to 9999 so they remain in the end after sort

    X -= jnp.floor(X)
    idx = jnp.lexsort((X[:,2], X[:,1], X[:,0], W_temp))

    #assert jnp.allclose(W, W[idx])
    A = A[idx]
    X = X[idx]
    return A, X

def letter_to_number(letter):
    """
    'a' to 1 , 'b' to 2 , 'z' to 26, and 'A' to 27 
    """
    return ord(letter) - ord('a') + 1 if 'a' <= letter <= 'z' else 27 if letter == 'A' else None

def shuffle(key, data):
    """
    shuffle data along batch dimension
    """
    G, L, XYZ, A, W = data
    idx = jax.random.permutation(key, jnp.arange(len(L)))
    return G[idx], L[idx], XYZ[idx], A[idx], W[idx]
    
def process_one(cif, atom_types, wyck_types, n_max, tol=0.01):
    """
    # taken from https://anonymous.4open.science/r/DiffCSP-PP-8F0D/diffcsp/common/data_utils.py
    Process one cif string to get G, L, XYZ, A, W

    Args:
      cif: cif string
      atom_types: number of atom types
      wyck_types: number of wyckoff types
      n_max: maximum number of atoms in the unit cell
      tol: tolerance for pyxtal

    Returns:
      G: space group number
      L: lattice parameters
      XYZ: fractional coordinates
      A: atom types
      W: wyckoff letters
    """
    crystal = Structure.from_str(cif, fmt='cif')
    spga = SpacegroupAnalyzer(crystal, symprec=tol)
    crystal = spga.get_refined_structure()
    c = pyxtal()
    try:
        c.from_seed(crystal, tol=0.01)
    except:
        c.from_seed(crystal, tol=0.0001)
    
    g = c.group.number
    num_sites = len(c.atom_sites)
    assert (n_max > num_sites) # we will need at least one empty site for output of L params

    print (g, c.group.symbol, num_sites)
    natoms = 0
    ww = []
    aa = []
    fc = []
    ws = []
    for site in c.atom_sites:
        a = element_list.index(site.specie) 
        x = site.position
        m = site.wp.multiplicity
        w = letter_to_number(site.wp.letter)
        symbol = str(m) + site.wp.letter
        natoms += site.wp.multiplicity
        assert (a < atom_types)
        assert (w < wyck_types)
        assert (np.allclose(x, site.wp[0].operate(x)))
        aa.append( a )
        ww.append( w )
        fc.append( x )  # the generator of the orbit
        ws.append( symbol )
        print ('g, a, w, m, symbol, x:', g, a, w, m, symbol, x)
    idx = np.argsort(ww)
    ww = np.array(ww)[idx]
    aa = np.array(aa)[idx]
    fc = np.array(fc)[idx].reshape(num_sites, 3)
    ws = np.array(ws)[idx]
    print (ws, aa, ww, natoms) 

    aa = np.concatenate([aa,
                        np.full((n_max - num_sites, ), 0)],
                        axis=0)

    ww = np.concatenate([ww,
                        np.full((n_max - num_sites, ), 0)],
                        axis=0)
    fc = np.concatenate([fc, 
                         np.full((n_max - num_sites, 3), 1e10)],
                        axis=0)
    
    abc = np.array([c.lattice.a, c.lattice.b, c.lattice.c])/natoms**(1./3.)
    angles = np.array([c.lattice.alpha, c.lattice.beta, c.lattice.gamma])
    l = np.concatenate([abc, angles])
    
    print ('===================================')

    return g, l, fc, aa, ww 

def process_one_c2db(file_path, atom_types, wyck_types, n_max, tol=0.1):
    crystal = read(file_path + '/structure.xyz')
    la = get_layergroup((crystal.get_cell(), crystal.get_scaled_positions(), crystal.get_atomic_numbers()), symprec=tol)
    g = la['number']
    l = la['std_lattice']
    map_ = la['mapping_to_primitive']
    std_map = la['std_mapping_to_primitive']
    std_pos = la['std_positions']
    std_types = la['std_types']
    std_chemical_symbols = np.array([element_list[i] for i in std_types])
    input_wyckoffs = la['wyckoffs']
    input_wyckoffs_num = [letter_to_number(w) for w in input_wyckoffs]

    assert all([type_ < atom_types for type_ in std_types])
    assert all([w < wyck_types for w in input_wyckoffs_num])

    primitive_equivalent_idx = la['equivalent_atoms']

    mult_list = []
    aa = []
    atom_species_list = []
    fc = []
    ww = []
    wyckoff_symbol_list = []
    
    for idx in set(primitive_equivalent_idx):
        full_idx, = np.where(primitive_equivalent_idx == idx)
        mult = len(full_idx)
        std_idx, = np.where(std_map == map_[idx])
        atom_type = list(set(std_types[std_idx]))
        atom_species = list(set(std_chemical_symbols[std_idx]))
        positions = std_pos[std_idx]
        wyckoff_letter = list(set(np.array(input_wyckoffs)[full_idx]))
        wyckoff_num = list(set(np.array(input_wyckoffs_num)[full_idx]))

        assert (len(atom_species) == 1)
        assert (len(wyckoff_letter) == 1)

        mult_list.append(mult)
        aa.append(atom_type[0])
        atom_species_list.append(atom_species[0])
        fc.append(positions[0])
        wyckoff_symbol = str(mult) + wyckoff_letter[0]
        wyckoff_symbol_list.append(wyckoff_symbol)
        ww.append(wyckoff_num[0])

        print ('g, a, m, symbol, x:', g, atom_species[0], mult, wyckoff_num[0], wyckoff_symbol, positions)

    ordered_idx = np.argsort(ww)
    ww = np.array(ww)[ordered_idx]
    aa = np.array(aa)[ordered_idx]
    fc = np.array(fc)[ordered_idx]
    wyckoff_symbol_list = np.array(wyckoff_symbol_list)[ordered_idx]
    mult_list = np.array(mult_list)[ordered_idx]
    atom_species_list = np.array(atom_species_list)[ordered_idx]
    
    natoms = sum(mult_list)
    l_pyxtal = xLattice.from_matrix(l)
    abc = np.array([l_pyxtal.a, l_pyxtal.b, l_pyxtal.c]) / natoms**(1./3.)
    angles = np.array([l_pyxtal.alpha, l_pyxtal.beta, l_pyxtal.gamma])
    l = np.concatenate([abc, angles])

    print(wyckoff_symbol_list, mult_list, atom_species_list, natoms)

    num_sites = len(fc)
    aa = np.concatenate([aa,
                        np.full((n_max - num_sites, ), 0)],
                        axis=0)

    ww = np.concatenate([ww,
                        np.full((n_max - num_sites, ), 0)],
                        axis=0)
    fc = np.concatenate([fc, 
                         np.full((n_max - num_sites, 3), 1e10)],
                        axis=0)

    print ('===================================')

    return g, l, fc, aa, ww




def GLXYZAW_from_file(sym_group, file_path, atom_types, wyck_types, n_max, num_workers=1):
    """
    Read cif strings from csv file and convert them to G, L, XYZ, A, W
    Note that cif strings must be in the column 'cif'

    Args:
      sym_group: SpaceGroup() or LayerGroup()
      file_path: path to the dataset
      atom_types: number of atom types
      wyck_types: number of wyckoff types
      n_max: maximum number of atoms in the unit cell
      num_workers: number of workers for multiprocessing

    Returns:
      G: space group number
      L: lattice parameters
      XYZ: fractional coordinates
      A: atom types
      W: wyckoff letters
    """
    if type(sym_group)==type(SpaceGroup()):
        data = pd.read_csv(file_path)
        cif_strings = data['cif']
        # print(type(cif_strings))

        p = multiprocessing.Pool(num_workers)
        partial_process_one = partial(process_one, atom_types=atom_types, wyck_types=wyck_types, n_max=n_max)
        results = p.map_async(partial_process_one, cif_strings).get()
        p.close()
        p.join()
    elif type(sym_group)==type(LayerGroup()):
        paths = os.walk(file_path)
        data_list = []
        for path, _, file_list in paths:
            for file_name in file_list:
                if file_name == 'data.json':
                    data_list.append(os.path.join(path, file_name).replace('/data.json', ''))
        
        p = multiprocessing.Pool(num_workers)
        partial_process_one_c2db = partial(process_one_c2db, atom_types=atom_types, wyck_types=wyck_types, n_max=n_max)
        results = p.map_async(partial_process_one_c2db, data_list).get()
        p.close()
        p.join()

    G, L, XYZ, A, W = zip(*results)

    G = jnp.array(G) 
    A = jnp.array(A).reshape(-1, n_max)
    W = jnp.array(W).reshape(-1, n_max)
    XYZ = jnp.array(XYZ).reshape(-1, n_max, 3)
    L = jnp.array(L).reshape(-1, 6)

    A, XYZ = sort_atoms(W, A, XYZ)
    
    return G, L, XYZ, A, W

def GLXA_to_structure_single(G, L, X, A):
    """
    Convert G, L, X, A to pymatgen structure. Do not use this function due to the bug in pymatgen.

    Args:
      G: space group number
      L: lattice parameters
      X: fractional coordinates
      A: atom types
    
    Returns:
      structure: pymatgen structure
    """
    lattice = Lattice.from_parameters(*L)
    # filter out padding atoms
    idx = np.where(A > 0)
    A = A[idx]
    X = X[idx]
    structure = Structure.from_spacegroup(sg=G, lattice=lattice, species=A, coords=X).as_dict()

    return structure

def GLXA_to_csv(G, L, X, A, num_worker=1, filename='out_structure.csv'):

    L = np.array(L)
    X = np.array(X)
    A = np.array(A)
    p = multiprocessing.Pool(num_worker)
    if isinstance(G, int):
        G = np.array([G] * len(L))
    structures = p.starmap_async(GLXA_to_structure_single, zip(G, L, X, A)).get()
    p.close()
    p.join()

    data = pd.DataFrame()
    data['cif'] = structures
    header = False if os.path.exists(filename) else True
    data.to_csv(filename, mode='a', index=False, header=header)


if __name__=='__main__':
    # atom_types = 119
    # wyck_types = 28
    # n_max = 24

    # import numpy as np
    # from crystalformer.src.sym_group import *
    # np.set_printoptions(threshold=np.inf)
    
    # #csv_file = '../data/mini.csv'
    # #csv_file = '/home/wanglei/cdvae/data/carbon_24/val.csv'
    # #csv_file = '/home/wanglei/cdvae/data/perov_5/val.csv'
    # csv_file = './mp_20/test_layer_test.csv'

    # G, L, XYZ, A, W = GLXYZAW_from_file(SpaceGroup(), csv_file, atom_types, wyck_types, n_max)
    
    # print (G.shape)
    # print (L.shape)
    # print (XYZ.shape)
    # print (A.shape)
    # print (W.shape)
    
    # print ('L:\n',L)
    # print ('XYZ:\n',XYZ)


    # @jax.vmap
    # def lookup(G, W):
    #     return SpaceGroup().mult_table[G-1, W] # (n_max, )
    # M = lookup(G, W) # (batchsize, n_max)
    # print ('N:\n', M.sum(axis=-1))
    atom_types = 119
    wyck_types = 18
    n_max = 27
    csv_file = './c2db'

    # process_one_c2db('./c2db/A/2C/1', atom_types, wyck_types, n_max)
    G, L, XYZ, A, W = GLXYZAW_from_file(LayerGroup(), csv_file, atom_types, wyck_types, n_max)
    
    print (G.shape)
    print (L.shape)
    print (XYZ.shape)
    print (A.shape)
    print (W.shape)

    print (G.shape)
    print (L.shape)
    print (XYZ.shape)
    print (A.shape)
    print (W.shape)
    
    print ('L:\n',L)
    print ('XYZ:\n',XYZ)


    @jax.vmap
    def lookup(G, W):
        return LayerGroup().mult_table[G-1, W] # (n_max, )
    M = lookup(G, W) # (batchsize, n_max)
    print ('N:\n', M.sum(axis=-1))
