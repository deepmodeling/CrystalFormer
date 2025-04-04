from config import *
from crystalformer.src.sample import project_xyz


def test_project_xyz():
    g = 225
    x = jnp.array([0.3, 0.5, 0.5])

    project_x = project_xyz(g, 1, x, 0)  # the 4a wyckoff position (0, 0, 0)
    assert jnp.allclose(project_x, jnp.array([0, 0, 0]))

    project_x = project_xyz(g, 6, x, 0)  # the 32f wyckoff position (x, x, x)
    assert jnp.allclose(project_x, jnp.array([0.3, 0.3, 0.3]))


test_project_xyz()
