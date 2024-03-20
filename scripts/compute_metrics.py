# Taken from: https://github.com/txie-93/cdvae/blob/main/scripts/compute_metrics.py
from collections import Counter
import argparse
import json
import os
import pandas as pd
from ast import literal_eval

import numpy as np
import multiprocessing
from pathlib import Path

from pymatgen.core.structure import Structure, Composition
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty

from eval_utils import (
    smact_validity, structure_validity)

# TODO: AttributeError in CrystalNNFP
CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset('magpie')


class Crystal(object):

    def __init__(self, crys_dict):
        self.crys_dict = crys_dict

        self.get_structure()
        self.get_composition()
        self.get_validity()
        # self.get_fingerprints()

    def get_structure(self):
        try:
            self.structure = Structure.from_dict(self.crys_dict)
            self.atom_types = [s.specie.number for s in self.structure]
            self.constructed = True
        except Exception:
            self.constructed = False
            self.invalid_reason = 'construction_raises_exception'
        if self.structure.volume < 0.1:
            self.constructed = False
            self.invalid_reason = 'unrealistically_small_lattice'

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype('int').tolist())

    def get_validity(self):
        self.comp_valid = smact_validity(self.elems, self.comps)
        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

    def get_fingerprints(self):
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = CompFP.featurize(comp)
        try:
            site_fps = [CrystalNNFP.featurize(
                self.structure, i) for i in range(len(self.structure))]
        except Exception:
            # counts crystal as invalid if fingerprint cannot be constructed.
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
        self.struct_fp = np.array(site_fps).mean(axis=0)


def get_validity(crys):
    comp_valid = np.array([c.comp_valid for c in crys]).mean()
    struct_valid = np.array([c.struct_valid for c in crys]).mean()
    valid = np.array([c.valid for c in crys]).mean()
    return {'comp_valid': comp_valid,
            'struct_valid': struct_valid,
            'valid': valid}

def get_crystal(cif_dict):
    try: return Crystal(cif_dict)
    except:
        print("Crystal construction failed")
        # print(cif_dict)
        struct = Structure.from_dict(cif_dict)
        print(struct)
        return None # return None if Crystal construction fails

def main(args):
    all_metrics = {}

    csv_path = os.path.join(args.root_path, args.filename)
    data = pd.read_csv(csv_path)
    cif_strings = data['cif']

    p = multiprocessing.Pool(args.num_io_process)
    crys_dict = p.map_async(literal_eval, cif_strings).get()
    # crys = p.map_async(Crystal, crys_dict).get()
    crys = p.map_async(get_crystal, crys_dict).get()
    crys = [c for c in crys if c is not None]
    print(f"Number of valid crystals: {len(crys)}")
    p.close()
    p.join()

    all_metrics['validity'] = get_validity(crys)
    print(all_metrics)

    if args.label == '':
        metrics_out_file = 'eval_metrics.json'
    else:
        metrics_out_file = f'eval_metrics_{args.label}.json'
    metrics_out_file = os.path.join(args.root_path, metrics_out_file)
    print("output path:", metrics_out_file)

    # only overwrite metrics computed in the new run.
    if Path(metrics_out_file).exists():
        with open(metrics_out_file, 'r') as f:
            written_metrics = json.load(f)
            if isinstance(written_metrics, dict):
                written_metrics.update(all_metrics)
            else:
                with open(metrics_out_file, 'w') as f:
                    json.dump(all_metrics, f)
        if isinstance(written_metrics, dict):
            with open(metrics_out_file, 'w') as f:
                json.dump(written_metrics, f)
    else:
        with open(metrics_out_file, 'w') as f:
            json.dump(all_metrics, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path',  default='/data/zdcao/crystal_gpt/dataset/mp_20/symm_data/')
    parser.add_argument('--filename', default='out_structure.csv')
    parser.add_argument('--label', default='')
    parser.add_argument('--num_io_process', type=int, default=40, help='number of process used in multiprocessing io')
    args = parser.parse_args()
    main(args)

