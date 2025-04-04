import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt 
import os 
from functools import partial

import sys
sys.path.append("./crystalformer/src/")
import checkpoint
from elements import element_list 

@partial(jax.vmap, in_axes=(0, None), out_axes=0) 
@partial(jax.vmap, in_axes=(None, 0), out_axes=0) 
def cosine_similarity(vector1, vector2):
    dot_product = jnp.dot(vector1, vector2)
    norm_a = jnp.linalg.norm(vector1)
    norm_b = jnp.linalg.norm(vector2)
    return dot_product / (norm_a * norm_b)

import argparse
parser = argparse.ArgumentParser(description="pretrain rdf")
parser.add_argument("--restore_path", default="/data/wanglei/crystalgpt/mp-mpsort-xyz-embed/w-a-x-y-z-periodic-fixed-size-embed-eb630/adam_bs_100_lr_0.0001_decay_0_clip_1_A_119_W_28_N_21_a_1_w_1_l_1_Nf_5_Kx_16_Kl_4_h0_256_l_8_H_8_k_32_m_64_e_32_drop_0.3/", help="")
args = parser.parse_args()

path = os.path.dirname(args.restore_path)


ckpt_filename, epoch_finished = checkpoint.find_ckpt_filename(args.restore_path) 
print("Load checkpoint file: %s, epoch finished: %g" %(ckpt_filename, epoch_finished))
ckpt = checkpoint.load_data(ckpt_filename)

a_embeddings = ckpt["params"]["~"]["a_embeddings"]
a_a = cosine_similarity(a_embeddings, a_embeddings)

g_embeddings = ckpt["params"]["~"]["g_embeddings"]
g_g = cosine_similarity(g_embeddings, g_embeddings)

print (a_a.shape, g_g.shape)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

ax = axes[0]
a_max = 90
ticks = np.arange(a_max)
element_ticks = [element_list[i+1] for i in ticks]
ax.set_xticks(ticks, labels=element_ticks, fontsize=8, rotation=90)
ax.set_yticks(ticks, labels=element_ticks, fontsize=8)
cax = ax.imshow(a_a[1:a_max+1, 1:a_max+1], cmap='coolwarm', interpolation='none')
fig.colorbar(cax, ax=ax) 

ax = axes[1]
cax = ax.imshow(g_g[:100, :100], cmap='coolwarm', interpolation='none')
fig.colorbar(cax, ax=ax) 

plt.show()
