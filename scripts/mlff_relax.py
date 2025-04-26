import warnings
# To suppress warnings for clearer output
warnings.simplefilter("ignore")

import pandas as pd
import os
from time import time
from ast import literal_eval
from tqdm import tqdm

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from ase.constraints import FixSymmetry


def make_orb_calc(model_path, device="cuda"):
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator

    # Load the ORB forcefield model
    orbff = pretrained.orb_v2(model_path, device=device) 
    calc = ORBCalculator(orbff, device=device)

    return calc


def make_matgl_calc(model_path, device="cuda"):
    import matgl
    from matgl.ext.ase import PESCalculator

    pot = matgl.load_model(model_path, device=device)
    calc = PESCalculator(pot)

    return calc


def make_mace_calc(model_path, device="cuda", default_dtype="float64"):
    from mace.calculators import mace_mp

    calc = mace_mp(model=model_path,
                   dispersion=False,
                   default_dtype=default_dtype,
                   device=device)
    return calc


def make_dp_calc(model_path):
    from deepmd.calculator import DP

    calc = DP(model_path)
    return calc


def relax_structures(calc, structures, relaxation, fmax, steps, fixsymmetry):
    """
    Args:
        calc: ASE calculator object
        structures: List of pymatgen Structure objects
        relaxation: Boolean, whether to relax the structures
        fmax: Maximum force tolerance for relaxation
        steps: Maximum number of steps for relaxation
    
    Returns:
        initial_energies: List of initial energies
        final_energies: List of final energies
        relaxed_cif_strings: List of relaxed structures in CIF format
        formula_list: List of formulas of the structures

    if relaxation is False, the final energies will be the same as the initial energies
    """

    ase_adaptor = AseAtomsAdaptor()

    initial_energies = []
    final_energies = []
    relaxed_cif_strings = []

    for i in tqdm(range(len(structures))):
        struct = structures[i]
        atoms = ase_adaptor.get_atoms(struct)
        atoms.calc = calc

        initial_energy = atoms.get_potential_energy()
        initial_energies.append(initial_energy)

        if relaxation:
            if fixsymmetry:
                try:
                    # Fix the space group symmetry of the structure
                    c = FixSymmetry(atoms) 
                    atoms.set_constraint(c)
                except Exception as e:
                    # sometimes the FixSymmetry constraint may not work if atoms are too close
                    print(f"Error fixing symmetry: {e}")

            # The following code is adapted from matgl repo
            # https://github.com/materialsvirtuallab/matgl/blob/824c1c4cefa9129c0af7066523d1665515f42899/src/matgl/ext/ase.py#L218-L304
            # Relax the structure using the FIRE optimizer
            optimizer = FrechetCellFilter(atoms)
            FIRE(optimizer).run(fmax=fmax, steps=steps)  # Run the FIRE optimizer for 100 steps
            
            final_energies.append(atoms.get_potential_energy())
            relaxed_cif_strings.append(ase_adaptor.get_structure(atoms).as_dict())

        else:
            final_energies.append(initial_energy)
            relaxed_cif_strings.append(struct.as_dict())


    formula_list = [struct.composition.formula for struct in structures]

    return initial_energies, final_energies, relaxed_cif_strings, formula_list


def main(args):
    csv_file = os.path.join(args.restore_path, args.filename)

    data = pd.read_csv(csv_file)
    cif_strings = data['cif']

    try: structures = [Structure.from_dict(literal_eval(cif)) for cif in cif_strings]
    except: structures = [Structure.from_str(cif, fmt="cif") for cif in cif_strings]

    if args.primitive:
        print("Converting structures to primitive form...")
        structures = [struct.get_primitive_structure() for struct in structures]

    print("Relaxing structures...")
    if args.relaxation:
        print("Relaxation is enabled. This may take a while.")
    else:
        print("Relaxation is disabled. Only initial energies will be calculated.")

    if args.fixsymmetry:
        print("Fixing space group symmetry of the structures.")

    print(f"Using {args.model} model at {args.model_path}")
    if args.model == "orb":
        calc = make_orb_calc(args.model_path, args.device)
    elif args.model == "matgl":
        calc = make_matgl_calc(args.model_path, args.device)
    elif args.model == "mace":
        calc = make_mace_calc(args.model_path, args.device)
    elif args.model == "dp":
        calc = make_dp_calc(args.model_path)
    else:
        raise ValueError("Invalid model type. Please choose from 'orb', 'matgl', 'mace' or 'dp'.")

    print("Calculating energies...")
    start_time = time()
    results  = relax_structures(calc, structures, args.relaxation, args.fmax, args.steps, args.fixsymmetry)
    end_time = time()
    print(f"Relaxation took {end_time - start_time:.2f} seconds")

    initial_energies, final_energies, relaxed_cif_strings, formula_list = results
    output_data = pd.DataFrame()
    output_data['initial_energy'] = initial_energies
    output_data['final_energy'] = final_energies
    output_data['relaxed_cif'] = relaxed_cif_strings
    output_data['formula'] = formula_list

    if args.label:
        output_data.to_csv(os.path.join(args.restore_path, f"relaxed_structures_{args.label}.csv"),
                           index=False)
    else:
        output_data.to_csv(os.path.join(args.restore_path, "relaxed_structures.csv"),
                           index=False)


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["orb", "matgl", "mace", "dp"], default="orb", help="choose the MLFF model")
    parser.add_argument("--device", type=str, default="cuda", help="choose the device to run the model on")
    parser.add_argument("--model_path", type=str, default="./data/orb-v2-20241011.ckpt", help="path to the model checkpoint")
    parser.add_argument("--restore_path", type=str, default="./experimental/", help="")
    parser.add_argument('--filename', default='output_struct.csv', help='filename of the csv file containing the structures')
    parser.add_argument('--relaxation', action='store_true', help='whether to relax the structures')
    parser.add_argument('--fmax', type=float, default=0.1, help='maximum force tolerance for relaxation')
    parser.add_argument('--steps', type=int, default=500, help='max number of steps for relaxation')
    parser.add_argument('--label', default=None, help='label for the output file')
    parser.add_argument('--primitive', action='store_true', help='convert structures to primitive form')
    parser.add_argument('--fixsymmetry', action='store_true', help='fix space group symmetry of the structures')

    args = parser.parse_args()
    main(args)
