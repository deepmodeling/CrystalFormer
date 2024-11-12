import sys
sys.path.append('../../crystalformer')

from abc import ABC, abstractmethod
import jax.numpy as jnp
from crystalformer.src.von_mises import von_mises_logpdf, sample_von_mises
from crystalformer.src.von_mises import gaussian_logpdf, sample_gaussian

class SymGroup(ABC):

    @abstractmethod
    def axis_name_assert(self, axis):
        pass

    @abstractmethod
    def distribution(self, axis):
        pass

    @abstractmethod
    def sample(self, axis):
        pass

class SpaceGroup(SymGroup):
    # sample of x,y,z should all be von mises
    def __init__(self):
        pass
    
    def axis_name_assert(self, axis):
        assert axis in {'x', 'y', 'z'}, "input error, axis needs to be 'x', 'y', or 'z'."
        return None

    def distribution(self, axis):
        self.axis_name_assert(axis)
        return von_mises_logpdf
    
    def sample(self, axis):
        self.axis_name_assert(axis)
        return sample_von_mises
    
    def sample_all_dim(self):
        return sample_von_mises

# class LayerGroup(SymGroup):
#     # sample of x,y should be von mises; sample of z should be gaussian
#     def __init__(self):
#         pass

#     def distribution(self, axis):
#         assert axis in {'x', 'y', 'z'}, "input error, axis needs to be 'x', 'y', or 'z'."

#         if axis == 'z':
#             return gaussian_logpdf
#         else:
#             return von_mises_logpdf




if __name__ == '__main__':
    a = SpaceGroup()
    a.distribution('x')
    a.sample('x')