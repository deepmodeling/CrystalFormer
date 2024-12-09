import sys
sys.path.append('./src/')

import pandas as pd
import numpy as np
from ast import literal_eval
import multiprocessing
import itertools
import argparse

from pymatgen.core import Structure, Lattice
from pymatgen.io.xyz import XYZ
from crystalformer.src.sym_group import LayerGroup

sym_group = LayerGroup()

symops = np.array(sym_group.symops)
mult_table = np.array(sym_group.mult_table)
wmax_table = np.array(sym_group.wmax_table)


def symmetrize_atoms(g, w, x):
    '''
    symmetrize atoms via, apply all sg symmetry op, finding the generator, and lastly apply symops 
    we need to do that because the sampled atom might not be at the first WP
    Args:
       g: int 
       w: int
       x: (3,)
    Returns:
       xs: (m, 3) symmetrize atom positions
    '''

    # (1) apply all space group symmetry op to the x 
    w_max = wmax_table[g-1].item()
    m_max = mult_table[g-1, w_max].item()
    ops = symops[g-1, w_max, :m_max] # (m_max, 3, 4)
    affine_point = np.array([*x, 1]) # (4, )
    coords = ops@affine_point # (m_max, 3) 
    coords -= np.floor(coords)

    # (2) search for the generator which satisfies op0(x) = x , i.e. the first Wyckoff position 
    # here we solve it in a jit friendly way by looking for the minimal distance solution for the lhs and rhs  
    #https://github.com/qzhu2017/PyXtal/blob/82e7d0eac1965c2713179eeda26a60cace06afc8/pyxtal/wyckoff_site.py#L115
    def dist_to_op0x(coord):
        diff = np.dot(symops[g-1, w, 0], np.array([*coord, 1])) - coord
        diff -= np.rint(diff)
        return np.sum(diff**2) 
   #  loc = np.argmin(jax.vmap(dist_to_op0x)(coords))
    loc = np.argmin([dist_to_op0x(coord) for coord in coords])
    x = coords[loc].reshape(3,)

    # (3) lastly, apply the given symmetry op to x
    m = mult_table[g-1, w] 
    ops = symops[g-1, w, :m]   # (m, 3, 4)
    affine_point = np.array([*x, 1]) # (4, )
    xs = ops@affine_point # (m, 3)
    xs -= np.floor(xs) # wrap back to 0-1 
    return xs

def get_struct_from_lawx(G, L, A, W, X):
    """
    Get the pymatgen.Structure object from the input data

    Args:
        G: space group number
        L: lattice parameters
        A: element number list
        W: wyckoff letter list
        X: fractional coordinates list
    
    Returns:
        struct: pymatgen.Structure object
    """
    A = A[np.nonzero(A)]
    X = X[np.nonzero(A)]
    W = W[np.nonzero(A)]


    xs_list = [symmetrize_atoms(G, w, x) for w, x in zip(W, X)]
    as_list = [[A[idx] for _ in range(len(xs))] for idx, xs in enumerate(xs_list)]
    A_list = list(itertools.chain.from_iterable(as_list))
    X_list = list(itertools.chain.from_iterable(xs_list))
    X_list += np.array([[0., 0., 0.5] for _ in range(len(X_list))])
    X_list -= np.floor(X_list)
    lattice = Lattice.from_parameters(*L)
    struct = Structure(lattice, A_list, X_list)
    return struct

def main(args):
    input_path = args.output_path + f'output_{args.label}.csv'
    origin_data = pd.read_csv(input_path)
    L,X,A,W = origin_data['L'],origin_data['X'],origin_data['A'],origin_data['W']
    L = L.apply(lambda x: literal_eval(x))
    X = X.apply(lambda x: literal_eval(x))
    A = A.apply(lambda x: literal_eval(x))
    W = W.apply(lambda x: literal_eval(x))
    # M = M.apply(lambda x: literal_eval(x))

    # convert array of list to numpy ndarray
    L = np.array(L.tolist())
    X = np.array(X.tolist())
    A = np.array(A.tolist())
    W = np.array(W.tolist())
    print(L.shape,X.shape,A.shape,W.shape)

    ### Multiprocessing. Use it if only run on CPU
    p = multiprocessing.Pool(args.num_io_process)
    G = np.array([int(args.label) for _ in range(len(L))])
    structures = p.starmap_async(get_struct_from_lawx, zip(G, L, A, W, X)).get()
    p.close()
    p.join()

    output_path = args.output_path + f'output_{args.label}_struct.csv'

    # data = pd.DataFrame()
    # data['cif'] = structures
    # data.to_csv(output_path, mode='a', index=False, header=True)
    i = 0
    for structure in structures:
        xyz = XYZ(structure)
        xyz.write_file(f'./{i}.xyz')
        i += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output_path', default='./', help='filepath of the output and input file')
    parser.add_argument('--label', default='194', help='output file label')
    parser.add_argument('--num_io_process', type=int, default=40, help='number of process used in multiprocessing io')
    args = parser.parse_args()
    main(args)
    
    # xs = symmetrize_atoms(61, 4, [ 0., 0., -0.04737133])
    # print(xs)
    # L = [3.829, 3.829, 40.172, 90.0, 90.0, 90.0]
    # A = [28, 8, 38, 17, 38, 8]
    # W = [4, 6, 5, 4, 2, 1]
    # X = [[ 0., 0., -0.04737133], [0., 0.5, 0.05508901], [0.5, 0.5, 0.09639814], [0., 0., -0.12889217], [0.5, 0.5, 0.], [0., 0., 0.]]
    # structure = get_struct_from_lawx(61, np.array(L), np.array(A), np.array(W), np.array(X))
    # print(structure)