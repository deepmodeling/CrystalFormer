import sys
sys.path.append('./crystalformer/src/')

import pandas as pd
import numpy as np
from ast import literal_eval
import multiprocessing
import itertools
import argparse

from pymatgen.core import Structure, Lattice
from wyckoff import wmax_table, mult_table, symops

symops = np.array(symops)
mult_table = np.array(mult_table)
wmax_table = np.array(wmax_table)


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

    lattice = Lattice.from_parameters(*L)
    xs_list = [symmetrize_atoms(G, w, x) for w, x in zip(W, X)]
    as_list = [[A[idx] for _ in range(len(xs))] for idx, xs in enumerate(xs_list)]
    A_list = list(itertools.chain.from_iterable(as_list))
    X_list = list(itertools.chain.from_iterable(xs_list))
    struct = Structure(lattice, A_list, X_list)
    return struct.as_dict()


def main(args):
    if args.label is not None:
        input_path = args.output_path + f'output_{args.label}.csv'
        output_path = args.output_path + f'output_{args.label}_struct.csv'
    else:
        input_path = args.output_path + f'output.csv'
        output_path = args.output_path + f'output_struct.csv'

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

    if args.label is None:
        G = origin_data['G']
        G = np.array(G.tolist())
    else:
        G = np.array([int(args.label) for _ in range(len(L))])

    ### Multiprocessing. Use it if only run on CPU
    p = multiprocessing.Pool(args.num_io_process)
    structures = p.starmap_async(get_struct_from_lawx, zip(G, L, A, W, X)).get()
    p.close()
    p.join()

    data = pd.DataFrame()
    data['cif'] = structures
    data.to_csv(output_path, mode='a', index=False, header=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output_path', default='./', help='filepath of the output and input file')
    parser.add_argument('--label', default=None, help='output file label')
    parser.add_argument('--num_io_process', type=int, default=40, help='number of process used in multiprocessing io')
    args = parser.parse_args()
    main(args)
