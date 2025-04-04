import pandas as pd
import argparse

from pymatgen.core import Structure, Composition
from pymatgen.analysis.bond_valence  import BVAnalyzer
from pymatgen.analysis.structure_prediction.volume_predictor import DLSVolumePredictor
from pymatgen.transformations.advanced_transformations import SubstitutionPredictorTransformation


def main(args):
    # initialize pymatgen objects
    bv_analyzer = BVAnalyzer(symm_tol=0)
    volume_predictor = DLSVolumePredictor()
    elem_sub = SubstitutionPredictorTransformation()

    data = pd.read_csv(args.input_path)

    # select data with spacegroup 
    data = data[data['spacegroup.number']==args.spacegroup]

    # select data with anonymized formula
    cif_strings = []
    for formula, cif in zip(data['pretty_formula'], data['cif']) :
        comp = Composition(formula)
        if comp.anonymized_formula == args.anonymized_formula:
            # print(formula)
            cif_strings.append(cif)
    print(len(cif_strings))

    structures = [Structure.from_str(cif, fmt='cif') for cif in cif_strings]

    oxi_structs = []

    for struct in structures:
        try:
            struct = bv_analyzer.get_oxi_state_decorated_structure(struct)
            oxi_structs.append(struct)
        except:
            print(struct.composition.reduced_formula)

    print(len(oxi_structs))

    sub_scale_structs = []
    prob_list = []
    for idx, struct in enumerate(oxi_structs):
        print(idx)
        try:
            sub_structs = elem_sub.apply_transformation(struct , return_ranked_list=10) # return top 3 structures
            print(f"there is {len(sub_structs)} sub structures")
            for _sub_struct in sub_structs:

                if _sub_struct['probability'] < float(args.prob_threshold):
                    continue
                
                sub_struct = _sub_struct['structure']
                if sub_struct.matches(struct):
                    continue
                
                scale_struct = volume_predictor.get_predicted_structure(sub_struct)
                sub_scale_structs.append(scale_struct)
                prob_list.append(_sub_struct['probability'])
        except Exception as e:
            print(e)
            # print(struct)
            continue

    print(len(sub_scale_structs))

    # remove duplicate structures in the list
    last_struct = []
    last_prob = []
    for idx, (struct, prob)  in enumerate(zip(sub_scale_structs, prob_list)):
        if idx == 0:
            last_struct.append(struct)
            last_prob.append(prob)
            continue

        if struct in last_struct:
            continue
        last_struct.append(struct)
        last_prob.append(prob)
    print(len(last_struct))

    # convert to conventional cell
    last_struct = [s.to_conventional() for s in last_struct]

    output_data = pd.DataFrame()
    output_data['cif'] = [struct.as_dict() for struct in last_struct]
    output_data['probability'] = last_prob
    output_data = output_data.sort_values(by='probability', ascending=False)
    # only select top num structures
    output_data = output_data.head(int(args.top_num))
    output_data.to_csv(args.output_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_path', default='/data/zdcao/crystal_gpt/dataset/mp_20/train.csv', help='filepath of the input file')
    parser.add_argument('--spacegroup', default=216, help='spacegroup number')
    parser.add_argument('--anonymized_formula', default='AB', help='anonymized formula')
    parser.add_argument('--prob_threshold', default=0.05, help='probability threshold')
    parser.add_argument('--top_num', default=100, help='top number of the output')
    parser.add_argument('--output_path', default='./test.csv', help='filepath of the output file')
    args = parser.parse_args()
    main(args)
