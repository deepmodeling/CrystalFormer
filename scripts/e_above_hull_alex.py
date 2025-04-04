import os
import json, bz2
import tempfile
import pandas as pd
import multiprocessing as mp
from functools import partial
from pymatgen.core import Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram
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


def calculate_hull(structure, energy, entries):
    entries = [ComputedStructureEntry.from_dict(i) for i in entries]
    pd = PhaseDiagram(entries)

    try:
        entry = generate_CSE(structure, energy)
        ehull = pd.get_e_above_hull(entry, allow_negative=True)
        print(f"Structure: {structure.formula}, E_hull: {ehull:.3f} eV/atom")
    except Exception as e:
        print(f"Structure: {structure.formula}, E_hull: Error: {e}")
        ehull = None
    
    return ehull


def forward_fn(structure, energy, ref_data):
    
    comp = structure.composition
    elements = set(ii.name for ii in comp.elements)

    # filter entries by elements
    entries = [entry for entry in ref_data['entries'] if set(entry['data']['elements']) <= elements]
    ehull = calculate_hull(structure, energy, entries)

    return ehull


def main(args):
    with bz2.open(args.convex_path) as fh:
        ref_data = json.loads(fh.read().decode('utf-8'))
    partial_forward_fn = partial(forward_fn, ref_data=ref_data)

    data = pd.read_csv(os.path.join(args.restore_path, args.filename))
    try: structures = [Structure.from_dict(eval(cif)) for cif in data['relaxed_cif']]
    except: structures = [Structure.from_str(cif, fmt="cif") for cif in data['relaxed_cif']]

    # with mp.Pool(args.num_io_process) as p:
    #     unrelaxed_ehull_list = p.map_async(partial_forward_fn, zip(structures, data['initial_energy'])).get()
    unrelaxed_ehull_list = list(map(partial_forward_fn, structures, data['initial_energy']))
    data['unrelaxed_ehull'] = unrelaxed_ehull_list

    if args.relaxation:
        # with mp.Pool(args.num_io_process) as p:
        #     relaxed_ehull_list = p.map_async(partial_forward_fn, zip(structures, data['final_energy'])).get()
        relaxed_ehull_list = list(map(partial_forward_fn, structures, data['final_energy']))
        data['relaxed_ehull'] = relaxed_ehull_list

    else:
        data['relaxed_ehull'] = unrelaxed_ehull_list    # same as unrelaxed

    if args.label:
        data.to_csv(f"{args.restore_path}/relaxed_structures_{args.label}_ehull.csv", index=False)
    else:
        data.to_csv(f"{args.restore_path}/relaxed_structures_ehull.csv", index=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Calculate e_above_hull for relaxed structures")
    parser.add_argument("--convex_path", type=str, default="/data/zdcao/crystal_gpt/dataset/alex/PBE/convex_hull_pbe_2023.12.29.json.bz2")
    parser.add_argument("--restore_path", type=str, default="./experimental/")
    parser.add_argument('--filename', default='relaxed_structures.csv')
    parser.add_argument('--relaxation', action='store_true')
    parser.add_argument('--label', default=None)
    parser.add_argument('--num_io_process', type=int, default=4)
    args = parser.parse_args()
    main(args)
