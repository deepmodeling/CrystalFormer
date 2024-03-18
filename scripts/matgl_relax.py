import matgl
from matgl.ext.ase import PESCalculator, Relaxer

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

import torch
torch.set_default_device("cuda")

import warnings
# To suppress warnings for clearer output
warnings.simplefilter("ignore")

import pandas as pd
import os
from time import time
from ast import literal_eval


def relax_structures(pot, structures, relaxations):
    """
    Relax structures using M3GNet potential

    Args:
      pot: M3GNet potential
      structures: list of pymatgen.Structure objects
      relaxations: boolean, whether to perform relaxation

    Returns:
        initial_energies: list of initial energies of the structures
        final_energies: list of final energies of the structures
        relaxed_cif_strings: list of relaxed structures in cif format
    """
    if relaxations:
        print("Relaxing structures with M3GNet...")
        relaxer = Relaxer(potential=pot)
        relax_results_list = [relaxer.relax(struct, fmax=0.01) for struct in structures]
        initial_energies = [relax_results["trajectory"].energies[0] for relax_results in relax_results_list]
        final_energies = [relax_results["trajectory"].energies[-1] for relax_results in relax_results_list]
        relaxed_cif_strings = [relax_results["final_structure"].to(fmt="cif") for relax_results in relax_results_list]
    else:
        print("No relaxation was performed. Returning initial energies as final energies.")
        ase_adaptor = AseAtomsAdaptor()
        initial_energies = []
        for struct in structures:
            # Create ase atom object
            atoms = ase_adaptor.get_atoms(struct)
            # define the M3GNet calculator
            calc = PESCalculator(pot)
            # set up the calculator for atoms object
            atoms.set_calculator(calc)
            initial_energies.append(atoms.get_potential_energy())
        final_energies = initial_energies    # if no relaxation, final energy is the same as initial energy
        relaxed_cif_strings = [struct.to(fmt="cif") for struct in structures]

    return initial_energies, final_energies, relaxed_cif_strings


def main(args):
    csv_file = os.path.join(args.restore_path, args.filename)

    data = pd.read_csv(csv_file)
    cif_strings = data['cif']

    try: structures = [Structure.from_dict(literal_eval(cif)) for cif in cif_strings]
    except: structures = [Structure.from_str(cif, fmt="cif") for cif in cif_strings]
    pot = matgl.load_model(args.model_path)
    print("Relaxing structures...")
    start_time = time()
    initial_energies, final_energies, relaxed_cif_strings  = relax_structures(pot, structures, args.relaxation)
    end_time = time()
    print(f"Relaxation took {end_time - start_time:.2f} seconds")

    output_data = pd.DataFrame()
    output_data['initial_energy'] = initial_energies
    output_data['final_energy'] = final_energies
    output_data['relaxed_cif'] = relaxed_cif_strings
    if args.label:
        output_data.to_csv(os.path.join(args.restore_path, f"relaxed_structures_{args.label}.csv"),
                           index=False)
    else:
        output_data.to_csv(os.path.join(args.restore_path, "relaxed_structures.csv"),
                           index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_path", type=str, default="/data/zdcao/crystal_gpt/data/b38199e3/adam_bs_100_lr_0.0001_decay_0_clip_1_A_119_W_28_N_21_a_1_w_1_l_1_Nf_5_Kx_16_Kl_4_h0_256_l_16_H_16_k_64_m_64_e_32_drop_0.5/unconditional/sorted")
    parser.add_argument('--filename', default='output_225_struct.csv')
    parser.add_argument('--relaxation', action='store_true')
    parser.add_argument('--model_path', default='/data/zdcao/website/matgl/pretrained_models/M3GNet-MP-2021.2.8-PES')
    parser.add_argument('--label', default='')
    args = parser.parse_args()
    main(args)
