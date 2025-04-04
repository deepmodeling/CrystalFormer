import os
import pandas as pd

from pymatgen.core import Structure, Composition
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def make_compare_structures(StructureMatcher):
    """
    Args:
        StructureMatcher: pymatgen.analysis.structure_matcher.StructureMatcher
    
    Returns:
        compare_structures: function, compare two structures
    """
    
    def compare_structures(s1, s2):
        """
        Args:
            s1: pymatgen Structure
            s2: pymatgen Structure
        
        Returns:
            bool, True if the two structures are the same
        """
    
        if s1.composition.reduced_composition != s2.composition.reduced_composition:
            return False
        else:
            return StructureMatcher.fit(s1, s2)

    return compare_structures


def make_search_duplicate(ref_data, StructureMatcher, spg_search=False):
    """
    Args:
        ref_data: pd.DataFrame, reference data
        StructureMatcher: pymatgen.analysis.structure_matcher.StructureMatcher
        spg_search: bool, whether to filter the reference data by space group number

    Returns:
        search_duplicate: function, search for duplicates in the reference data
    """

    def search_duplicate(s):
        """
        Args:
            s: pymatgen Structure

        Returns:
            duplicate: bool, True if the structure is a duplicate

        sometimes the matching of the space group number is not accurate
        so we will not use it to filter the reference data
        """

        if spg_search:
            try:
                spg_analyzer = SpacegroupAnalyzer(s)
                spg = spg_analyzer.get_space_group_number()

            except Exception as e:
                spg = None
                print(e)
                print(f"Error with structure {s}")
                pass

            if spg is not None:
                sub_data = ref_data[ref_data['spg'] == spg]
            else: 
                sub_data = ref_data

        else:
            sub_data = ref_data
    
        # pick all structures with the same composition
        sub_data = sub_data[sub_data['composition'] == s.composition.reduced_composition]

        duplicate = False
        # compare the structure with all structures with the same composition
        for s2 in sub_data['structure']:
            s2 = Structure.from_dict(eval(s2))
            if StructureMatcher.fit(s, s2):
                duplicate = True
                break

        return duplicate
    
    return search_duplicate


def main(args):

    # print all the parameters
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    data = pd.read_csv(os.path.join(args.restore_path, args.filename))
    ref_data = pd.read_csv(args.ref_path)

    # only keep the necessary columns
    ref_data = ref_data[['formula', 'elements', 'structure', 'spg']]

    if args.spg_search and args.spg is not None:
        ref_data = ref_data[ref_data['spg'] == args.spg]  # filter by space group
        print(f"Number of structures in the reference data with space group {args.spg}: {ref_data.shape[0]}")
    else:
        print(f"Number of structures in the reference data: {ref_data.shape[0]}")

    sm = StructureMatcher()
    compare_structures = make_compare_structures(sm)

    # remove unstable structures
    data = data[data['relaxed_ehull'] <= 0.1]
    structures = [Structure.from_dict(eval(crys_dict)) for crys_dict in data['relaxed_cif']]
    print(f"Number of stable structures: {len(structures)}")

    # remove duplicates (Uniqueness)
    idx_list = []
    unique_structures = []
    for idx, s in enumerate(structures):
        if not any([compare_structures(s, us) for us in unique_structures]):
            unique_structures.append(s)
            idx_list.append(idx)

    data = data.iloc[idx_list]
    print(f"Number of stable and unique structures: {len(unique_structures)}")

    # remove structures that are already in the reference data (Novelty)
    comp_list = []
    for idx, formula in enumerate(ref_data['formula']):
        try:
            comp = Composition(formula)
            comp_list.append(comp)
        except Exception as e:
            # Can't parse formula when formula is NaN
            print(e)
            print(f"Error with formula {formula}")
            if ref_data.iloc[idx]['elements'] == "['Na', 'N']":
                comp_list.append(Composition("NaN"))

    print(len(comp_list))
    comp_list = [comp.reduced_composition for comp in comp_list]
    ref_data['composition'] = comp_list

    search_duplicate = make_search_duplicate(ref_data, sm, args.spg_search)
    duplicate_list = list(map(search_duplicate, unique_structures))

    # pick the idx of False in duplicate_list
    idx_list = [idx for idx, duplicate in enumerate(duplicate_list) if not duplicate]
    data = data.iloc[idx_list]
    print(f"Number of stable, unique and novel structures: {data.shape[0]}")

    if args.spg is not None:
        data.to_csv(os.path.join(args.restore_path, f"sun_structures_{args.spg}.csv"), index=False)
    else:
        data.to_csv(os.path.join(args.restore_path, "sun_structures.csv"), index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Check the stable, Unique and Novelty structures")
    parser.add_argument("--spg", type=int, default=None, help="Space group number")
    parser.add_argument("--spg_search", action="store_true", help="Whether to filter the reference data by space group number")
    parser.add_argument("--restore_path", type=str, default=None, help="Path to the restored data")
    parser.add_argument("--filename", type=str, default="relaxed_structures_ehull.csv", help="Filename of the restored data")
    parser.add_argument("--ref_path", type=str, default="/data/zdcao/crystal_gpt/dataset/alex/PBE/alex20/alex20.csv", help="Path to the reference data")
    args = parser.parse_args()
    main(args)
