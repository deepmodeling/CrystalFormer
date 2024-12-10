from abc import ABC, abstractmethod
from crystalformer.src.von_mises import sample_von_mises

class SymGroup(ABC):
    @abstractmethod
    def sample(self):
        pass

class SpaceGroup(SymGroup):
    def __init__(self):
        pass

    def sample(self, key, loc, concentration, shape):
        return sample_von_mises(key, loc, concentration, shape)

class LayerGroup(SymGroup):
    def __init__(self):
        pass

    def sample(self, key, loc, concentration, shape):
        samples = gaussian_centered(key, concentration, shape)
        samples += loc
        return samples

