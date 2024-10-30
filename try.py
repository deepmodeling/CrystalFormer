import sys
sys.path.append('./crystalformer/src/')

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from hydra import initialize, compose


import checkpoint
from transformer import make_transformer

with initialize(version_base=None, config_path="./model"):
    args = compose(config_name="config")
    print(args)

key = jax.random.PRNGKey(42)
params, transformer = make_transformer(key, args.Nf, args.Kx, args.Kl, args.n_max,
                                        args.h0_size,
                                        4, 8,
                                        32, args.model_size, args.embed_size,
                                        args.atom_types, args.wyck_types,
                                        0.3)


print("\n========== Load checkpoint==========")
restore_path = "./share/"
ckpt_filename, epoch_finished = checkpoint.find_ckpt_filename(restore_path)
if ckpt_filename is not None:
    print("Load checkpoint file: %s, epoch finished: %g" %(ckpt_filename, epoch_finished))
    ckpt = checkpoint.load_data(ckpt_filename)
    params = ckpt["params"]
else:
    print("No checkpoint file found. Start from scratch.")

print ("# of transformer params", ravel_pytree(params)[0].size)



import numpy as np
from pymatgen.core import Structure, Lattice
from time import time
from pymatgen.io.ase import AseAtomsAdaptor
from ase.visualize import view

from sample import sample_crystal
from elements import element_dict, element_list
from scripts.awl2struct import get_struct_from_lawx

jax.config.update("jax_enable_x64", True) # to get off compilation warning, and to prevent sample nan lattice


def generate_and_visualize(spacegroup, elements, temperature, seed):

  print(f"Generating with spacegroup={spacegroup}, elements={elements}, temperature={temperature}")
  
  top_p = 1
  n_sample = 1
  elements = elements.split()
  if elements is not None:
      idx = [element_dict[e] for e in elements]
      atom_mask = [1] + [1 if a in idx else 0 for a in range(1, args.atom_types)]
      atom_mask = jnp.array(atom_mask)
      # print ('sampling structure formed by these elements:', elements)
    #   print (atom_mask)
    #   print("@")
  else:
      atom_mask = jnp.zeros((args.atom_types), dtype=int) # we will do nothing to a_logit in sampling
      print (atom_mask)
  
  # fix
  atom_mask = jnp.repeat(atom_mask.reshape(1, -1), args.n_max, axis=0)
  key = jax.random.PRNGKey(seed)
  key, subkey = jax.random.split(key)
  start_time = time()
#   import pdb
#   pdb.set_trace()
  XYZ, A, W, M, L = sample_crystal(subkey, transformer, params, args.n_max, n_sample, args.atom_types, args.wyck_types, args.Kx, args.Kl, spacegroup, None, atom_mask, top_p, temperature, temperature, jnp.repeat(args.use_foriloop, args.n_max))
  end_time = time()
  print("executation time:", end_time - start_time)
  
  XYZ = np.array(XYZ)
  A = np.array(A)
  W = np.array(W)
  L = np.array(L)
  
  G = np.array([spacegroup for i in range(len(L))])
  
  structures = [get_struct_from_lawx(g, l, a, w, xyz) for g, l, a, w, xyz in zip(G, L, A, W, XYZ)]
  structures = [Structure.from_dict(_) for _ in structures]
  
  
  atoms_list = [AseAtomsAdaptor().get_atoms(struct) for struct in structures]
  return view(atoms_list[0], viewer='ngl')




# ============= params to control the generation =============
spacegroup = 225   # 设置生成的晶体的空间群，范围为1-230
elements = "Ba Ti O"      # 限制生成晶体所包含的元素种类，每个元素需要用空格隔开，比如 "Ba Ti O"
temperature = 1.0  # 控制transformer生成的温度，温度越高生成的novelty越高，推荐值为 0.5到1.5
seed = 42          # 随机种子

# =============== generate and visualization =================
generate_and_visualize(spacegroup, elements, temperature, seed)