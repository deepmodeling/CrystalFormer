from abc import ABC, abstractmethod
import jax.numpy as jnp
from crystalformer.src.von_mises import von_mises_logpdf

class SymGroup(ABC):
    @abstractmethod
    def x_distribution(self):
        pass

class SpaceGroup(SymGroup):
    # sample of x,y,z should all be von mises
    def __init__(self):
        pass

    def x_distribution(self):
        return von_mises_logpdf
    
    def y_distribution(self):
        return von_mises_logpdf
    
    def z_distribution(self):
        return von_mises_logpdf

class LayerGroup(SymGroup):
    # sample of x,y should be von mises; sample of z should be gaussian
    def __init__(self):
        pass

    def x_distribution(x, loc, concentration):
        return von_mises_logpdf(x, loc, concentration)
    
    def y_distribution(x, loc, concentration):
        return von_mises_logpdf(x, loc, concentration)
    
    def z_distribution(x, loc, concentration):
        # concentration = 1/sigma^2
        return -0.5 * (jnp.log(2 * jnp.pi) - jnp.log(concentration) + concentration * (x - loc) * (x - loc))