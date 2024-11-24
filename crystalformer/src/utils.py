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
import json
import re

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

def process_one_c2db(data, atom_types, wyck_types, n_max, tol=0.01):
    # sym_group = LayerGroup()
    # data_file = open(data_dir + '/data.json')
    # data = json.load(data_file)
    g = data['lgnum']

    # def lattice_from_c2db(file_path):
    #     file = open(file_path, 'r')
    #     lines = file.readlines()
    #     lattice_str = re.search(r'Lattice="([^"]+)"', lines[1]).group(1)
    #     lattice_list = [float(v) for v in lattice_str.split()]

    #     a_vec = lattice_list[:3]
    #     b_vec = lattice_list[3:6]
    #     c_vec = lattice_list[6:]

    #     a_length = np.linalg.norm(a_vec)
    #     b_length = np.linalg.norm(b_vec)
    #     c_length = np.linalg.norm(c_vec)

    #     alpha = np.arccos(np.dot(b_vec, c_vec) / (b_length * c_length))
    #     beta = np.arccos(np.dot(a_vec, c_vec) / (a_length * c_length))
    #     gamma = np.arccos(np.dot(a_vec, b_vec) / (a_length * b_length))

    #     return np.array([a_length, b_length, c_length, alpha, beta, gamma])

    # def check_wyckoff_symbol(coord, g):
    #     for i in range(1,20):
    #         transformed_coords = sym_group.symops[g, i] @ jnp.array([*coord,1])
    #         for t_coord in transformed_coords:
    #             if bool(jnp.allclose(jnp.array(t_coord), jnp.array(coord), atol=tol)):
    #                 return i

    # def atom_info(file_path):
    #     f = open(file_path, 'r')
    #     lines = f.readlines()
    #     lattice_str = re.search(r'Lattice="([^"]+)"', lines[1]).group(1)
    #     lattice_list = [float(v) for v in lattice_str.split()]
    #     lattice_mat = jnp.reshape(jnp.array(lattice_list),(3,3))

    #     atoms = []
    #     for atom in lines[2:]:
    #         atom = atom.split()
    #         for i in range(1,4):
    #             atom[i] = float(atom[i])
    #         frac_coord = jnp.linalg.inv(lattice_mat) @ jnp.array(atom[1:4])
    #         for i in range(1,4):
    #             atom[i] = float(frac_coord[i-1])
    #         atoms.append(atom[0:4])
    #     return atoms

    # l = lattice_from_c2db(data_dir + '/structure.xyz')
    l = eval(data['l'])

    # atoms = atom_info(data_dir + '/structure.xyz')
    atoms = eval(data['atoms'])
    pos = eval(data['positions'])
    wyckoff = eval(data['wyckoff'])
    num_sites = len(atoms)
    print(g, num_sites)

    natoms = 0
    ww = []
    aa = []
    fc = []
    ws = []
    for i in range(num_sites):
        try:
            a = element_list.index(atoms[i])
        except:
            print(i)
            print(data)
        x = pos[i]
        symbol = wyckoff[i]
        w = letter_to_number(symbol[-1])
        m = int(symbol[:-1])
        natoms += m
        assert (a < atom_types)
        assert (w < wyck_types)
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
    print ('===================================')
    return g, l, fc, aa, ww 
    






def GLXYZAW_from_file(sym_group, csv_file, atom_types, wyck_types, n_max, num_workers=1):
    """
    Read cif strings from csv file and convert them to G, L, XYZ, A, W
    Note that cif strings must be in the column 'cif'

    Args:
      csv_file: csv file containing cif strings
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
    data = pd.read_csv(csv_file)
    if type(sym_group)==type(SpaceGroup()):
        cif_strings = data['cif']
        # print(type(cif_strings))

        p = multiprocessing.Pool(num_workers)
        partial_process_one = partial(process_one, atom_types=atom_types, wyck_types=wyck_types, n_max=n_max)
        results = p.map_async(partial_process_one, cif_strings).get()
        p.close()
        p.join()
    elif type(sym_group)==type(LayerGroup()):
        data_list = []
        for i in range(len(data)):
            single_data = {'lgnum':data['LayerGroup'][i], 'l':data['Lattice'][i], 'atoms':data['Elements'][i], 'positions':data['FractionCoords'][i], 'wyckoff':data['WyckoffSymbols'][i]}
            data_list.append(single_data)
        # paths = os.walk(csv_file)
        # data_dir = []
        # for path, dir_list, file_list in paths:
        #     for file_name in file_list:
        #         if file_name == 'data.json':
        #             data_dir.append(os.path.join(path, file_name).replace('/data.json', ''))
        # print(data_dir)
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
    n_max = 24
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
