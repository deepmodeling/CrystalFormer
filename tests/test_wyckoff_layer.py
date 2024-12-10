wyckoff_list = [[1], [1, 1, 1, 1, 2], [1, 1, 1, 1, 2], [1, 2], [2], [1, 1, 1, 1, 2, 2, 2, 2, 2, 4], [2, 2, 2, 2, 4], [1, 1, 2], [2], [2, 4], [1, 1, 2], [2], [2, 4], [1, 1, 1, 1, 2, 2, 2, 2, 4], [2, 2, 2, 4], [2, 2, 2, 4], [2, 2, 4], [2, 2, 4, 4, 4, 8], [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 4], [2, 2, 2, 4], [2, 2, 4], [2, 2, 4, 4, 4, 4, 4, 8], [1, 1, 1, 1, 2, 2, 2, 2, 4], [2, 2, 2, 4], [2, 2, 4], [2, 2, 4, 4, 4, 8], [1, 1, 2, 2, 2, 4], [2, 2, 4], [2, 4], [2, 2, 4], [2, 2, 4], [2, 4], [4], [2, 4], [2, 4, 4, 8], [4, 4, 8], [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 8], [2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 8], [2, 2, 4, 4, 4, 4, 4, 8], [2, 2, 2, 4, 4, 4, 4, 8], [2, 2, 2, 2, 4, 4, 4, 4, 8], [2, 2, 4, 4, 4, 8], [4, 4, 4, 8], [2, 2, 4, 4, 4, 8], [4, 4, 4, 8], [2, 2, 4, 4, 4, 8], [2, 2, 4, 4, 4, 4, 4, 8, 8, 8, 8, 16], [4, 4, 4, 4, 8, 8, 8, 8, 8, 16], [1, 1, 2, 4], [1, 1, 2, 2, 2, 4], [1, 1, 2, 2, 2, 4, 4, 8], [2, 2, 4, 4, 8], [1, 1, 2, 2, 2, 4, 4, 4, 4, 8], [2, 2, 4, 4, 8], [1, 1, 2, 4, 4, 4, 8], [2, 2, 4, 8], [1, 1, 2, 2, 2, 4, 4, 4, 4, 8], [2, 2, 4, 4, 8], [1, 1, 2, 2, 2, 4, 4, 4, 8], [2, 2, 4, 4, 4, 8], [1, 1, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 16], [2, 2, 4, 4, 4, 8, 8, 8, 16], [2, 2, 4, 4, 4, 8, 8, 16], [2, 2, 4, 4, 8, 8, 8, 16], [1, 1, 1, 3], [1, 2, 2, 3, 6], [1, 1, 1, 2, 2, 2, 3, 6], [1, 2, 2, 3, 6], [1, 1, 1, 3, 6], [1, 2, 3, 6], [1, 2, 2, 3, 4, 6, 6, 12], [1, 2, 2, 3, 6, 6, 12], [1, 2, 3, 6], [1, 1, 1, 2, 2, 2, 3, 6], [1, 2, 2, 3, 4, 6, 6, 12], [1, 2, 2, 3, 4, 6, 6, 6, 12], [1, 2, 3, 6, 6, 12], [1, 1, 1, 2, 2, 2, 3, 6, 6, 12], [1, 2, 2, 3, 4, 6, 6, 12], [1, 2, 2, 3, 4, 6, 6, 6, 12, 12, 12, 24]]


# symbols = []

# for ws in wyckoff_list:
#     ws_symbols = []
#     for i in range(len(ws)):
#         ws_symbols.append(f'{ws[i]}{chr(i+97)}')
#     symbols.append(f"{ws_symbols}")
            

# # print(symbols)

# import pandas as pd
# # import numpy as np

# # data = np.array([[i for i in range(1, 1 + len(symbols))], symbols])
# df = pd.DataFrame(data={'Layer Group': [i for i in range(1, 1 + len(symbols))], 'Wyckoff Positions': symbols})
# df.to_csv('layer_symbols.csv', index=False)



from config import *
from crystalformer.src.sym_group import *

sym_group = LayerGroup()

def test_mult_table():

    def nonzero_part(arr):
        nonzero_indices = jnp.nonzero(arr)
        return arr[nonzero_indices]

    def match(g):
        jnp.allclose( nonzero_part(sym_group.mult_table[g-1]) , jnp.array(wyckoff_list[g-1]))

    match(25)
    match(47)
    # match(99)
    # match(123)
    # match(221)

def test_wyckoff():
    import pandas as pd
    import os

    df = pd.read_csv(os.path.join(datadir, 'layer_symbols.csv'))
    df['Wyckoff Positions'] = df['Wyckoff Positions'].apply(eval)  # convert string to list

    wyckoff_symbols = df['Wyckoff Positions'].tolist()

    import numpy as np
    import jax.numpy as jnp

    wyckoff_list = []
    wyckoff_dict = []
    for ws in wyckoff_symbols:
        wyckoff_list.append( [0] +[0 if w == "" else int(''.join(filter(str.isdigit, w))) for w in ws] )

        ws = [""] + ws
        wyckoff_dict.append( {value: index for index, value in enumerate(ws)} )

    max_len = max(len(sublist) for sublist in wyckoff_list)
    mult_table = np.zeros((len(wyckoff_list), max_len), dtype=int) # mult_table[g-1, w] = multiplicity 
    wmax_table = np.zeros((len(wyckoff_list),), dtype=int)   # wmax_table[g-1] = number of wyckoff letters 
    for i, sublist in enumerate(wyckoff_list):
        mult_table[i, :len(sublist)] = sublist
        wmax_table[i] = len(sublist)-1
    mult_table = jnp.array(mult_table)
    wmax_table = jnp.array(wmax_table)
    
    assert jnp.allclose(mult_table, sym_group.mult_table)
    assert jnp.allclose(wmax_table, sym_group.wmax_table)

# def test_symmetrize_atoms():
#     from crystalformer.src.wyckoff import symmetrize_atoms
#     from pymatgen.symmetry.groups import SpaceGroup

#     #https://github.com/materialsproject/pymatgen/blob/1e347c42c01a4e926e15b910cca8964c1a0cc826/pymatgen/symmetry/groups.py#L547
#     def in_array_list(array_list: list[np.ndarray], arr: np.ndarray, tol: float = 1e-5) -> bool:
#         """Extremely efficient nd-array comparison using numpy's broadcasting. This
#         function checks if a particular array a, is present in a list of arrays.
#         It works for arrays of any size, e.g., even matrix searches.

#         Args:
#             array_list ([array]): A list of arrays to compare to.
#             arr (array): The test array for comparison.
#             tol (float): The tolerance. Defaults to 1e-5. If 0, an exact match is done.

#         Returns:
#             (bool)
#         """
#         if len(array_list) == 0:
#             return False
#         axes = tuple(range(1, arr.ndim + 1))
#         if not tol:
#             return any(np.all(array_list == arr[None, :], axes))
#         return any(np.sum(np.abs(array_list - arr[None, :]), axes) < tol)

#     def symmetrize_atoms_deduplication(g, w, x):
#         '''
#         symmetrize atoms via deduplication
#         this implements the same method as pmg get_orbit function, see
#         #https://github.com/materialsproject/pymatgen/blob/1e347c42c01a4e926e15b910cca8964c1a0cc826/pymatgen/symmetry/groups.py#L328
#         Args:
#            g: int 
#            w: int
#            x: (3,)
#         Returns:
#            xs: (m, 3)  symmetrized atom positions
#         '''
#         # (1) apply all space group symmetry ops to x 
#         w_max = sym_group.wmax_table[g-1].item()
#         m_max = sym_group.mult_table[g-1, w_max].item()
#         ops = sym_group.symops[g-1, w_max, :m_max] # (m_max, 3, 4)
#         affine_point = jnp.array([*x, 1]) # (4, )
#         coords = ops@affine_point # (m_max, 3) 
        
#         # (2) deduplication to select the orbit 
#         orbit: list[np.ndarray] = []
#         for pp in coords:
#             pp = np.mod(np.round(pp, decimals=10), 1) # round and mod to avoid duplication
#             if not in_array_list(orbit, pp):
#                 orbit.append(pp)
#         orbit -= np.floor(orbit)   # wrap back to 0-1 
#         assert (orbit.shape[0] == sym_group.mult_table[g-1, w]) # double check that the orbit has the right length
#         return orbit

#     def symmetrize_atoms_pmg(g, w, x):
#         sg = SpaceGroup.from_int_number(g)
#         xs = sg.get_orbit(x)
#         m = sym_group.mult_table[g-1, w]  
#         assert (len(xs) == m) # double check that the orbit has the right length
#         return np.array(xs)

#     def allclose_up_to_permutation(xs, xs_pmg):
#         # Sort each array lexicographically by rows
#         sorted_xs = xs[np.lexsort(np.rot90(xs))]
#         sorted_xs_pmg = xs_pmg[np.lexsort(np.rot90(xs_pmg))]
#         # Check if the sorted arrays are equal
#         return np.allclose(sorted_xs, sorted_xs_pmg)
 
#     g = 166 
#     w = jnp.array(3)
#     x = jnp.array([0., 0., 0.5619])
#     xs = symmetrize_atoms(sym_group, g, w, x)
#     print ('xs:\n', xs)
#     assert allclose_up_to_permutation(xs, symmetrize_atoms_pmg(g, w, x))
#     assert allclose_up_to_permutation(xs, symmetrize_atoms_deduplication(g, w, x))

#     g = 225
#     w = jnp.array(5)
#     x = jnp.array([0., 0., 0.7334])
#     xs = symmetrize_atoms(sym_group, g, w, x)
#     print ('xs:\n', xs)
#     assert allclose_up_to_permutation(xs, symmetrize_atoms_pmg(g, w, x))
#     assert allclose_up_to_permutation(xs, symmetrize_atoms_deduplication(g, w, x))

#     g = 225
#     w = jnp.array(8)
#     x = jnp.array([0.0, 0.23, 0.23])
#     xs = symmetrize_atoms(sym_group, g, w, x)
#     print ('xs:\n', xs)
#     assert allclose_up_to_permutation(xs, symmetrize_atoms_pmg(g, w, x))
#     assert allclose_up_to_permutation(xs, symmetrize_atoms_deduplication(g, w, x))

# test_symmetrize_atoms()
test_wyckoff()