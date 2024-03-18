import os
import pandas as pd
import numpy as np
import tempfile
from mp_api.client import MPRester
from pymatgen.core import Structure
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.io.vasp.inputs import Incar, Poscar


# Taken from https://github.com/facebookresearch/crystal-llm/blob/main/e_above_hull.py
def generate_CSE(structure, m3gnet_energy):
    # Write VASP inputs files as if we were going to do a standard MP run
    # this is mainly necessary to get the right U values / etc
    b = MPRelaxSet(structure)
    with tempfile.TemporaryDirectory() as tmpdirname:
        b.write_input(f"{tmpdirname}/", potcar_spec=True)
        poscar = Poscar.from_file(f"{tmpdirname}/POSCAR")
        incar = Incar.from_file(f"{tmpdirname}/INCAR")
        clean_structure = Structure.from_file(f"{tmpdirname}/POSCAR")

    # Get the U values and figure out if we should have run a GGA+U calc
    param = {"hubbards": {}}
    if "LDAUU" in incar:
        param["hubbards"] = dict(zip(poscar.site_symbols, incar["LDAUU"]))
    param["is_hubbard"] = (
        incar.get("LDAU", True) and sum(param["hubbards"].values()) > 0
    )
    if param["is_hubbard"]:
        param["run_type"] = "GGA+U"

    # Make a ComputedStructureEntry without the correction
    cse_d = {
        "structure": clean_structure,
        "energy": m3gnet_energy,
        "correction": 0.0,
        "parameters": param,
    }

    # Apply the MP 2020 correction scheme (anion/+U/etc)
    cse = ComputedStructureEntry.from_dict(cse_d)
    _ = MaterialsProject2020Compatibility(check_potcar=False).process_entries(
        cse,
        clean=True,
    )

    # Return the final CSE (notice that the composition/etc is also clean, not things like Fe3+)!
    return cse


def get_strutures_ehull(mpr, structures, energies):
    """
    Get the e_above_hull for a list of structures

    Args:
        mpr: MPRester object
        structures: list of pymatgen.Structure objects
        energies: list of energies of the structures
    
    Returns:
        ehull_list: list of e_above_hull values
    """
    ehull_list = []
    for s, e in zip(structures, energies):
        # entry = PDEntry(s.composition, e)
        entry = generate_CSE(s, e)
        elements = [el.name for el in entry.composition.elements]

        # Obtain only corrected GGA and GGA+U ComputedStructureEntry objects
        entries = mpr.get_entries_in_chemsys(elements=elements, 
                                             additional_criteria={"thermo_types": ["GGA_GGA+U"],
                                                                  "is_stable": True}     # Only stable entries
                                            )
        pd = PhaseDiagram(entries)
        try:
            ehull = pd.get_e_above_hull(entry, allow_negative=True)
            ehull_list.append(ehull)
            print(f"Structure: {s.formula}, E_hull: {ehull:.3f} eV/atom")
        except:
            print(f"Structure: {s.formula}, E_hull: N/A")
            ehull_list.append(np.nan)

    return ehull_list


def main(args):
    data = pd.read_csv(os.path.join(args.restore_path, args.filename))
    cif_strings = data["relaxed_cif"]
    structures = [Structure.from_str(cif, fmt="cif") for cif in cif_strings]
    mpr = MPRester(args.api_key)

    unrelaxed_ehull_list = get_strutures_ehull(mpr, structures, data["initial_energy"])
    if args.relaxation:
        relaxed_ehull_list = get_strutures_ehull(mpr, structures, data["final_energy"])
    else:
        relaxed_ehull_list = [np.nan] * len(structures)  # Fill with NaNs

    output_data = pd.DataFrame()
    output_data["relaxed_cif"] = cif_strings
    output_data["relaxed_ehull"] = relaxed_ehull_list
    output_data["unrelaxed_ehull"] = unrelaxed_ehull_list
    if args.label:
        output_data.to_csv(f"ehull_{args.label}.csv", index=False)
    else:
        output_data.to_csv("ehull.csv", index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_path", type=str, default="/data/zdcao/crystal_gpt/dataset/mp_20/")
    parser.add_argument('--filename', default='relaxed_structures_testdata.csv')
    parser.add_argument('--relaxation', action='store_true')
    parser.add_argument('--api_key', default='9zBRHS6Zp94KE28PeMdSk5gCyteIm6Ks')
    parser.add_argument('--label', default='testdata')
    args = parser.parse_args()
    main(args)
