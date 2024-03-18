import os, sys
testdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(testdir, "../src"))

import jax
import jax.numpy as jnp
import numpy as np 
import haiku as hk
