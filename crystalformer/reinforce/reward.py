import jax
import jax.numpy as jnp
import numpy as np
import joblib
from pymatgen.core import Structure, Lattice

from crystalformer.reinforce import ehull
from crystalformer.src.wyckoff import wmax_table, mult_table, symops


symops = np.array(symops)
mult_table = np.array(mult_table)
wmax_table = np.array(wmax_table)


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
    affine_point = np.array([*x, 1]) # (4, )
    coords = ops@affine_point # (m_max, 3) 
    coords -= np.floor(coords)

    # (2) search for the generator which satisfies op0(x) = x , i.e. the first Wyckoff position 
    # here we solve it in a jit friendly way by looking for the minimal distance solution for the lhs and rhs  
    #https://github.com/qzhu2017/PyXtal/blob/82e7d0eac1965c2713179eeda26a60cace06afc8/pyxtal/wyckoff_site.py#L115
    def dist_to_op0x(coord):
        diff = np.dot(symops[g-1, w, 0], np.array([*coord, 1])) - coord
        diff -= np.rint(diff)
        return np.sum(diff**2) 
   #  loc = np.argmin(jax.vmap(dist_to_op0x)(coords))
    loc = np.argmin([dist_to_op0x(coord) for coord in coords])
    x = coords[loc].reshape(3,)

    # (3) lastly, apply the given symmetry op to x
    m = mult_table[g-1, w] 
    ops = symops[g-1, w, :m]   # (m, 3, 4)
    affine_point = np.array([*x, 1]) # (4, )
    xs = ops@affine_point # (m, 3)
    xs -= np.floor(xs) # wrap back to 0-1 
    return xs


def get_atoms_from_GLXYZAW(G, L, XYZ, A, W):

    A = A[np.nonzero(A)]
    X = XYZ[np.nonzero(A)]
    W = W[np.nonzero(A)]

    lattice = Lattice.from_parameters(*L)
    xs_list = [symmetrize_atoms(G, w, x) for w, x in zip(W, X)]
    A_list = np.repeat(A, [len(xs) for xs in xs_list])
    X_list = np.concatenate(xs_list)
    struct = Structure(lattice, A_list, X_list).to_ase_atoms()
    return struct


def make_force_reward_fn(calculator, weight=1.0):
    """
    Args:
        calculator: ase calculator object
        weight: weight for stress, total reward = log(forces + weight*stress)

    Returns:
        reward_fn: single reward function
        batch_reward_fn: batch reward function
    """
    def reward_fn(x):
        G, L, XYZ, A, W = x
        try: 
            atoms = get_atoms_from_GLXYZAW(G, L, XYZ, A, W)
            atoms.calc = calculator
            forces = atoms.get_forces()
            stress = atoms.get_stress()
        except: 
            forces = np.ones((1, 3))*np.inf # avoid nan
            stress = np.ones((6,))*np.inf
        forces = np.linalg.norm(forces, axis=-1)
        forces = np.clip(forces, 1e-2, 1e2)  # avoid too large or too small forces
        forces = np.mean(forces)
        stress = np.clip(np.abs(stress), 1e-2, 1e2)
        stress = np.mean(stress)
        
        return np.log(forces + weight*stress)

    def batch_reward_fn(x):
        x = jax.tree_util.tree_map(lambda _x: jax.device_put(_x, jax.devices('cpu')[0]), x)
        G, L, XYZ, A, W = x
        G, L, XYZ, A, W = np.array(G), np.array(L), np.array(XYZ), np.array(A), np.array(W)
        x = (G, L, XYZ, A, W)
        output = map(reward_fn, zip(*x))
        output = np.array(list(output))
        output = jax.device_put(output, jax.devices('gpu')[0]).block_until_ready()

        return output

    return reward_fn, batch_reward_fn


def make_ehull_reward_fn(calculator, ref_data, batch=50, n_jobs=-1):
    """
    Args:
        calculator: ase calculator object
        ref_data: reference data for ehull calculation

    Returns:
        reward_fn: single reward function
        batch_reward_fn: batch reward function
    """

    from pymatgen.io.ase import AseAtomsAdaptor

    ase_adaptor = AseAtomsAdaptor()

    def energy_fn(x):
        G, L, XYZ, A, W = x
        try: 
            atoms = get_atoms_from_GLXYZAW(G, L, XYZ, A, W)
            atoms.calc = calculator
            energy = atoms.get_potential_energy()
            structure = ase_adaptor.get_structure(atoms)
        except:
            energy = np.inf
            structure = None
        
        return structure, energy

    def reward_fn(structure, energy):
        if structure == None:
            e_above_hull = np.inf
        else:    
            try: 
                e_above_hull = ehull.forward_fn(structure, energy, ref_data)
            except:
                e_above_hull = np.inf

        # clip e above hull to avoid too large or too small values
        e_above_hull = np.clip(e_above_hull, -10, 10)

        return e_above_hull
    
    def map_reward_fn(structures, energies):
        output = map(reward_fn, structures, energies)

        return list(output)
    
    def parallel_reward_fn(structures, energies):
        xs = [(structures[i:i+batch], energies[i:i+batch]) for i in range(0, len(structures), batch)]
        output = joblib.Parallel(
                        n_jobs=n_jobs
                    )(joblib.delayed(map_reward_fn)(*x) for x in xs)
        # concatenate the output
        output = np.concatenate(output)

        return output

    def batch_reward_fn(x):
        x = jax.tree_util.tree_map(lambda _x: jax.device_put(_x, jax.devices('cpu')[0]), x)
        G, L, XYZ, A, W = x
        G, L, XYZ, A, W = np.array(G), np.array(L), np.array(XYZ), np.array(A), np.array(W)
        x = (G, L, XYZ, A, W)
        structures, energies = zip(*map(energy_fn, zip(*x)))
        output = parallel_reward_fn(structures, energies)
        output = jnp.array(output)
        output = jax.device_put(output, jax.devices('gpu')[0]).block_until_ready()

        return output

    return reward_fn, batch_reward_fn


def make_prop_reward_fn(model, target, dummy_value=5, loss_type='mse'):

    """
    Args:
        model: property prediction model, takes pymatgen structure as input, returns property value
        target: target property value
        dummy_value: dummy value to return if model fails to predict
        loss_type: loss function type, 'mse' or 'mae'

    Returns:
        reward_fn: single reward function
        batch_reward_fn: batch reward function

    """

    from pymatgen.io.ase import AseAtomsAdaptor

    ase_adaptor = AseAtomsAdaptor()

    def reward_fn(x):
        G, L, XYZ, A, W = x
        try: 
            atoms = get_atoms_from_GLXYZAW(G, L, XYZ, A, W)
            struct = ase_adaptor.get_structure(atoms)
            quantity = model(struct)
            # if quantity is nan, return a dummy value
            quantity = quantity if not np.isnan(quantity) else np.array(dummy_value)
        except:
            quantity = np.array(dummy_value)  #TODO: check if this is a good idea
        
        return quantity

    def batch_reward_fn(x):
        x = jax.tree_util.tree_map(lambda _x: jax.device_put(_x, jax.devices('cpu')[0]), x)
        G, L, XYZ, A, W = x
        G, L, XYZ, A, W = np.array(G), np.array(L), np.array(XYZ), np.array(A), np.array(W)
        x = (G, L, XYZ, A, W)
        output = map(reward_fn, zip(*x))
        output = jnp.array(list(output)) - target
        
        if loss_type == 'mae':
            output = jnp.abs(output)
        elif loss_type == 'mse':
            output = output**2  # MSE loss
        else:
            raise ValueError('Invalid loss type')
        
        output = jax.device_put(output, jax.devices('gpu')[0]).block_until_ready()

        return output

    return reward_fn, batch_reward_fn


def make_dielectric_reward_fn(models, dummy_value=0):
    """
    Reward function for dielectric reward. models contains two models, one for dielectric constant and one for band gap.
    the reward is the product of the two quantities.

    Args:
        models: list of property prediction models, each takes pymatgen structure as input, returns property value
        dummy_value: dummy value to return if model fails to predict

    Returns:
        reward_fn: single reward function
        batch_reward_fn: batch reward function
    """

    from pymatgen.io.ase import AseAtomsAdaptor
    from ase.stress import voigt_6_to_full_3x3_stress
    ase_adaptor = AseAtomsAdaptor()

    assert len(models) == 2, 'models should contain two models, one for dielectric constant and one for band gap'

    def reward_fn(x):
        G, L, XYZ, A, W = x
        try: 
            atoms = get_atoms_from_GLXYZAW(G, L, XYZ, A, W)
            struct = ase_adaptor.get_structure(atoms)
            
            pred = models[0](struct)
            dielectric_tensor = voigt_6_to_full_3x3_stress(pred)
            eigenvalues, _ = np.linalg.eig(dielectric_tensor)
            scalar_dielectric = np.mean(np.real(eigenvalues))
            if np.isnan(scalar_dielectric):
                return np.array(dummy_value)

            band_gap = models[1](struct).item()
            if np.isnan(band_gap):
                return np.array(dummy_value)
            
            reward = - np.array(scalar_dielectric * band_gap)

        except:
            reward = np.array(dummy_value)  #TODO: check if this is a good idea
        
        return reward


    def batch_reward_fn(x):
        x = jax.tree_util.tree_map(lambda _x: jax.device_put(_x, jax.devices('cpu')[0]), x)
        G, L, XYZ, A, W = x
        G, L, XYZ, A, W = np.array(G), np.array(L), np.array(XYZ), np.array(A), np.array(W)
        x = (G, L, XYZ, A, W)
        output = map(reward_fn, zip(*x))
        output = np.array(list(output))
        output = jax.device_put(output, jax.devices('gpu')[0]).block_until_ready()

        return output

    return reward_fn, batch_reward_fn
