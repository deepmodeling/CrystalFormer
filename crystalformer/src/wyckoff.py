import pandas as pd
import os
import numpy as np 
import re
import jax
import jax.numpy as jnp

def from_xyz_str(xyz_str: str):
    """
    Args:
        xyz_str: string of the form 'x, y, z', '-x, -y, z', '-2y+1/2, 3x+1/2, z-y+1/2', etc.
    Returns:
        affine operator as a 3x4 array
    """
    rot_matrix = np.zeros((3, 3))
    trans = np.zeros(3)
    tokens = xyz_str.strip().replace(" ", "").lower().split(",")
    re_rot = re.compile(r"([+-]?)([\d\.]*)/?([\d\.]*)([x-z])")
    re_trans = re.compile(r"([+-]?)([\d\.]+)/?([\d\.]*)(?![x-z])")
    for i, tok in enumerate(tokens):
        # build the rotation matrix
        for m in re_rot.finditer(tok):
            factor = -1.0 if m.group(1) == "-" else 1.0
            if m.group(2) != "":
                factor *= float(m.group(2)) / float(m.group(3)) if m.group(3) != "" else float(m.group(2))
            j = ord(m.group(4)) - 120
            rot_matrix[i, j] = factor
        # build the translation vector
        for m in re_trans.finditer(tok):
            factor = -1 if m.group(1) == "-" else 1
            num = float(m.group(2)) / float(m.group(3)) if m.group(3) != "" else float(m.group(2))
            trans[i] = num * factor
    return np.concatenate( [rot_matrix, trans[:, None]], axis=1) # (3, 4)


df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/wyckoff_list.csv'))
df['Wyckoff Positions'] = df['Wyckoff Positions'].apply(eval)  # convert string to list
wyckoff_positions = df['Wyckoff Positions'].tolist()

symops = np.zeros((230, 28, 576, 3, 4)) # 576 is the least common multiple for all possible mult
mult_table = np.zeros((230, 28), dtype=int) # mult_table[g-1, w] = multiplicity , 28 because we had pad 0 
wmax_table = np.zeros((230,), dtype=int)    # wmax_table[g-1] = number of possible wyckoff letters for g 
dof0_table = np.ones((230, 28), dtype=bool)  # dof0_table[g-1, w] = True for those wyckoff points with dof = 0 (no continuous dof)
fc_mask_table = np.zeros((230, 28, 3), dtype=bool) # fc_mask_table[g-1, w] = True for continuous fc 

def build_g_code():
    #use general wyckoff position as the code for space groups
    xyz_table = []
    g_table = []
    for g in range(230):
        wp0 = wyckoff_positions[g][0]
        g_table.append([])
        for xyz in wp0:
            if xyz not in xyz_table: 
                xyz_table.append(xyz)
            g_table[-1].append(xyz_table.index(xyz))
        assert len(g_table[-1]) == len(set(g_table[-1]))

    g_code = []
    for g in range(230):
        g_code.append( [1 if i in g_table[g] else 0 for i in range(len(xyz_table))] )
    del xyz_table
    del g_table
    g_code = jnp.array(g_code)
    return g_code

for g in range(230):
    wyckoffs = []
    for x in wyckoff_positions[g]:
        wyckoffs.append([])
        for y in x:
            wyckoffs[-1].append(from_xyz_str(y))
    wyckoffs = wyckoffs[::-1] # a-z,A

    mult = [len(w) for w in wyckoffs]
    mult_table[g, 1:len(mult)+1] = mult
    wmax_table[g] = len(mult)

    # print (g+1, [len(w) for w in wyckoffs])
    for w, wyckoff in enumerate(wyckoffs):
        wyckoff = np.array(wyckoff)
        repeats = symops.shape[2] // wyckoff.shape[0]
        symops[g, w+1, :, :, :] = np.tile(wyckoff, (repeats, 1, 1))
        dof0_table[g, w+1] = np.linalg.matrix_rank(wyckoff[0, :3, :3]) == 0
        fc_mask_table[g, w+1] = jnp.abs(wyckoff[0, :3, :3]).sum(axis=1)!=0 

symops = jnp.array(symops)
mult_table = jnp.array(mult_table)
wmax_table = jnp.array(wmax_table)
dof0_table = jnp.array(dof0_table)
fc_mask_table = jnp.array(fc_mask_table)

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
    affine_point = jnp.array([*x, 1]) # (4, )
    coords = ops@affine_point # (m_max, 3) 
    coords -= jnp.floor(coords)

    # (2) search for the generator which satisfies op0(x) = x , i.e. the first Wyckoff position 
    # here we solve it in a jit friendly way by looking for the minimal distance solution for the lhs and rhs  
    #https://github.com/qzhu2017/PyXtal/blob/82e7d0eac1965c2713179eeda26a60cace06afc8/pyxtal/wyckoff_site.py#L115
    def dist_to_op0x(coord):
        diff = jnp.dot(symops[g-1, w, 0], jnp.array([*coord, 1])) - coord
        diff -= jnp.rint(diff)
        return jnp.sum(diff**2) 
    loc = jnp.argmin(jax.vmap(dist_to_op0x)(coords))
    x = coords[loc].reshape(3,)

    # (3) lastly, apply the given symmetry op to x
    m = mult_table[g-1, w] 
    ops = symops[g-1, w, :m]   # (m, 3, 4)
    affine_point = jnp.array([*x, 1]) # (4, )
    xs = ops@affine_point # (m, 3)
    xs -= jnp.floor(xs) # wrap back to 0-1 
    return xs

if __name__=='__main__':
    print (symops.shape)
    print (symops.size*symops.dtype.itemsize//(1024*1024))

    import numpy as np 
    np.set_printoptions(threshold=np.inf)

    print (symops[166-1,3, :6])
    op = symops[166-1, 3, 0]
    print (op)
    
    w_max = wmax_table[225-1]
    m_max = mult_table[225-1, w_max]
    print ('w_max, m_max', w_max, m_max)

    print (fc_mask_table[225-1, 6])
    sys.exit(0)
    
    print ('mult_table')
    print (mult_table[25-1]) # space group id -> multiplicity table
    print (mult_table[42-1])
    print (mult_table[47-1])
    print (mult_table[99-1])
    print (mult_table[123-1])
    print (mult_table[221-1])
    print (mult_table[166-1])

    print ('dof0_table')
    print (dof0_table[25-1])
    print (dof0_table[42-1])
    print (dof0_table[47-1])
    print (dof0_table[225-1])
    print (dof0_table[166-1])
    
    print ('wmax_table')
    print (wmax_table[47-1])
    print (wmax_table[123-1])
    print (wmax_table[166-1])

    print ('wmax_table', wmax_table)
    
    atom_types = 119 
    aw_max = wmax_table*(atom_types-1)    # the maximum value of aw
    print ( (aw_max-1)%(atom_types-1)+1 ) # = 118 
    print ( (aw_max-1)//(atom_types-1)+1 ) # = wmax
