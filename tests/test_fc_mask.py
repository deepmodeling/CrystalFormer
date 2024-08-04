from config import *
import pandas as pd
import os
import numpy as np
import jax.numpy as jnp

df = pd.read_csv(os.path.join(datadir, 'wyckoff_list.csv'))
df['Wyckoff Positions'] = df['Wyckoff Positions'].apply(eval)  # convert string to list
wyckoff_positions = df['Wyckoff Positions'].tolist()

def convert_to_binary_list(s):
    """
    Converts a list of strings into a list of binary lists based on the presence of 'x', 'y', or 'z'.
    """
    components = s.split(',')
    #TODO a better translation can be xxx->100 but not 111 
    return  [1 if any(char in comp for char in ['x', 'y', 'z']) else 0 for comp in components]

fc_mask_list = []
for g, wp_list in enumerate(wyckoff_positions):
    sub_list = []
    for wp in wp_list[::-1]:
        sub_list.append(convert_to_binary_list(wp[0]))
    fc_mask_list.append(sub_list)

max_len = max(len(sub_list) for sub_list in fc_mask_list)

fc_mask_table = np.zeros((len(fc_mask_list), max_len+1, 3), dtype=int) # (230, 28, 3)
for i, sub_list in enumerate(fc_mask_list):
    for j, l in enumerate(sub_list):
        fc_mask_table[i, j+1, : ] = l    # we have added a padding of W=0
fc_mask_table = jnp.array(fc_mask_table) # 1 in the fc_mask_table select those active fractional coordinate  

from config import *

def test_fc_mask():
    from crystalformer.src.wyckoff import symops, wmax_table
    from crystalformer.src.wyckoff import fc_mask_table as fc_mask_table_test

    for g in range(1, 231):
        for w in range(1, wmax_table[g]+1):
            op = symops[g-1, w, 0] # 0 since we conly consider the first wyckoff point in the equivalent class when building fc_mask_table
            fc_mask = (op[:3, :3].sum(axis=1)!=0)
            assert jnp.allclose(fc_mask, fc_mask_table[g-1, w])
            assert jnp.allclose(fc_mask, fc_mask_table_test[g-1, w])

test_fc_mask() 
