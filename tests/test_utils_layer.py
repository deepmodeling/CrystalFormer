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

    paths = os.walk(csv_file)
    data_list = []
    for path, _, file_list in paths:
        for file_name in file_list:
            if file_name == 'data.json':
                data_list.append(os.path.join(path, file_name).replace('/data.json', ''))

    

    # G, L, X, A, W = process_one_c2db(csv_file, atom_types, mult_types, n_max)
    G, L, X, A, W = GLXYZAW_from_file(sym_group, csv_file, atom_types, mult_types, n_max, num_workers=128)

    error_list = []

    for i in range(len(data_list)):
        crystal = read(data_list[i] + '/structure.xyz')
        ase_adaptor = AseAtomsAdaptor()
        struct = ase_adaptor.get_structure(crystal)

        g = G[i]
        l = np.array(L[i])
        x = np.array(X[i])
        a = np.array(A[i])
        w = np.array(W[i])

        M = mult_table[g-1, w]
        num_atoms = np.sum(M, axis=-1)
        l[3:] = l[3:] * (180 / np.pi)
        l[:3] = l[:3] * (num_atoms)**(1./3.)
        
        struct_converted = get_struct_from_lawx(g, l, a, w, x)

        matcher = StructureMatcher(primitive_cell=False, attempt_supercell=True)
        isMatch = matcher.fit(struct, struct_converted)
        
        if not isMatch:
            error_list.append(data_list[i])

    return error_list

path = '/home/longlizheng/pycode/CrystalFormer_layer/c2db_stab'
result = test_utils(path)
print(result)