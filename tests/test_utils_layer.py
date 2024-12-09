from config import *

from ase import Atoms
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

from crystalformer.src.utils import GLXYZAW_from_file, process_one_c2db
from crystalformer.src.sym_group import *
from scripts.awl2struct_2d import get_struct_from_lawx

sym_group = LayerGroup()
mult_table = sym_group.mult_table

# def calc_n(G, W):
#     @jax.vmap
#     def lookup(G, W):
#         return mult_table[G-1, W] # (n_max, )
#     M = lookup(G, W) # (batchsize, n_max)
#     N = M.sum(axis=-1)
#     return N

def test_utils(csv_file):

    atom_types = 119
    mult_types = 18
    n_max = 27

    crystal = read(csv_file + '/structure.xyz')
    ase_adaptor = AseAtomsAdaptor()
    struct = ase_adaptor.get_structure(crystal)

    # G, L, X, A, W = process_one_c2db(csv_file, atom_types, mult_types, n_max)
    G, L, X, A, W = GLXYZAW_from_file(sym_group, csv_file, atom_types, mult_types, n_max)
    g = G[0]
    l = np.array(L[0])
    x = np.array(X[0])
    a = np.array(A[0])
    w = np.array(W[0])

    l[3:] = l[3:] * (180 / np.pi)
    l[:3] = l[:3] * (len(a))**(1./3.)
    
    struct_converted = get_struct_from_lawx(g, l, a, w, x)

    matcher = StructureMatcher()
    isMatch = matcher.fit(struct, struct_converted)

    return isMatch

if __name__ == '__main__':
    for i in range(4784):
        isMatch = test_utils(f'/Users/longli/Downloads/c2db_stab/{i}')
        assert isMatch == True, f"Mat No.{i} error."
