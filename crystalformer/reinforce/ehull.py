import tempfile
from pymatgen.core import Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry, ComputedEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.io.vasp.inputs import Incar, Poscar


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
    entries = [ComputedEntry.from_dict(i) for i in entries]
    pd = PhaseDiagram(entries)
    
    entry = generate_CSE(structure, energy)
    ehull = pd.get_e_above_hull(entry, allow_negative=True)
    
    return ehull


def forward_fn(structure, energy, ref_data):
    
    comp = structure.composition
    elements = set(ii.name for ii in comp.elements)

    # filter entries by elements
    entries = [entry for entry in ref_data['entries'] if set(entry['data']['elements']) <= elements]
    ehull = calculate_hull(structure, energy, entries)

    return ehull
