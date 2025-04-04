import numpy as np

from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import NeighborList
from ase.stress import full_3x3_to_voigt_6_stress


class ExponentialPotential(Calculator):
    """
    Exponential potential for ASE.
    u(r) = exp(-alpha * r)

    https://gitlab.com/ase/ase/-/blob/master/ase/calculators/lj.py?ref_type=heads
    """

    implemented_properties = ['energy', 'energies', 'forces', 'free_energy']
    implemented_properties += ['stress', 'stresses']  # bulk properties
    default_parameters = {
        'alpha': 1.0,
        'rc': None,
        'ro': None,
        'smooth': False,
    }
    nolabel = True

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        alpha: float
          The decay constant of the exponential potential, default 1.0
        rc: float, None
          Cut-off for the NeighborList. The energy is upshifted to be continuous at rc.
          Default None
        ro: float, None
          Onset of cutoff function in 'smooth' mode. Defaults to 0.66 * rc.
        smooth: bool, False
          Cutoff mode. False means that the pairwise energy is simply shifted
          to be 0 at r = rc, leading to the energy going to 0 continuously,
          but the forces jumping to zero discontinuously at the cutoff.
          True means that a smooth cutoff function is multiplied to the pairwise
          energy that smoothly goes to 0 between ro and rc. Both energy and
          forces are continuous in that case.
          If smooth=True, make sure to check the tail of the
          forces for kinks, ro might have to be adjusted to avoid distorting
          the potential too much.

        """

        Calculator.__init__(self, **kwargs)

        if self.parameters.rc is None:
            self.parameters.rc = 10.0  # Choose an appropriate rc for your system

        if self.parameters.ro is None:
            self.parameters.ro = 0.66 * self.parameters.rc

        self.nl = None

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)

        natoms = len(self.atoms)

        alpha = self.parameters.alpha
        rc = self.parameters.rc
        ro = self.parameters.ro
        smooth = self.parameters.smooth

        if self.nl is None or 'numbers' in system_changes:
            self.nl = NeighborList(
                [rc / 2] * natoms, self_interaction=False, bothways=True
            )

        self.nl.update(self.atoms)

        positions = self.atoms.positions
        cell = self.atoms.cell

        # potential value at rc
        e0 = np.exp(-alpha * rc)

        energies = np.zeros(natoms)
        forces = np.zeros((natoms, 3))
        stresses = np.zeros((natoms, 3, 3))

        for ii in range(natoms):
            neighbors, offsets = self.nl.get_neighbors(ii)
            cells = np.dot(offsets, cell)

            # pointing *towards* neighbours
            distance_vectors = positions[neighbors] + cells - positions[ii]

            r = np.sqrt((distance_vectors ** 2).sum(1))
            r[r > rc] = np.inf  # Exclude pairs beyond cutoff

            if smooth:
                cutoff_fn = cutoff_function(r ** 2, rc ** 2, ro ** 2)
                d_cutoff_fn = d_cutoff_function(r ** 2, rc ** 2, ro ** 2)

            pairwise_energies = np.exp(-alpha * r)
            pairwise_forces = -alpha * np.exp(-alpha * r) / r  # du_ij/dr

            if smooth:
                pairwise_forces = (
                    cutoff_fn * pairwise_forces + 2 * d_cutoff_fn
                    * pairwise_energies
                )
                pairwise_energies *= cutoff_fn
            else:
                pairwise_energies -= e0 * (r != 0.0)

            pairwise_forces = pairwise_forces[:, np.newaxis] * distance_vectors

            energies[ii] += 0.5 * pairwise_energies.sum()  # atomic energies
            forces[ii] += pairwise_forces.sum(axis=0)

            stresses[ii] += 0.5 * np.dot(
                pairwise_forces.T, distance_vectors
            )  # equivalent to outer product

        # no lattice, no stress
        if self.atoms.cell.rank == 3:
            stresses = full_3x3_to_voigt_6_stress(stresses)
            self.results['stress'] = stresses.sum(
                axis=0) / self.atoms.get_volume()
            self.results['stresses'] = stresses / self.atoms.get_volume()

        energy = energies.sum()
        self.results['energy'] = energy
        self.results['energies'] = energies

        self.results['free_energy'] = energy

        self.results['forces'] = forces


def cutoff_function(r, rc, ro):
    """Smooth cutoff function.

    Goes from 1 to 0 between ro and rc, ensuring
    that u(r) = exp(-alpha * r) * cutoff_function(r) is C^1.

    Defined as 1 below ro, 0 above rc.

    Note that r, rc, ro are all expected to be squared,
    i.e. `r = r_ij^2`, etc.

    """

    return np.where(
        r < ro,
        1.0,
        np.where(r < rc, (rc - r) ** 2 * (rc + 2 *
                 r - 3 * ro) / (rc - ro) ** 3, 0.0),
    )


def d_cutoff_function(r, rc, ro):
    """Derivative of smooth cutoff function wrt r.

    Note that `r = r_ij^2`, so for the derivative wrt to `r_ij`,
    we need to multiply `2*r_ij`. This gives rise to the factor 2
    above, the `r_ij` is cancelled out by the remaining derivative
    `d r_ij / d d_ij`, i.e. going from scalar distance to distance vector.
    """

    return np.where(
        r < ro,
        0.0,
        np.where(r < rc, 6 * (rc - r) * (ro - r) / (rc - ro) ** 3, 0.0),
    )
