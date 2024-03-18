# Function: compute the metrics of the generated structures
# It takes about 20 min to compute the metrics of 9000 generated structures for MP-20 dataset
import pandas as pd
from pymatgen.core import Structure
import multiprocessing
import argparse
from ast import literal_eval
import json
from time import time

from matbench_genmetrics.core.metrics import GenMetrics


def get_structure(cif):
    try:
        return Structure.from_str(cif, fmt='cif')
    except:
        return Structure.from_dict(literal_eval(cif))

def main(args):
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    gen_df = pd.read_csv(args.gen_path)

    p = multiprocessing.Pool(args.num_io_process)
    train_structures = p.map_async(get_structure, train_df['cif']).get()
    test_structures = p.map_async(get_structure, test_df['cif']).get()
    gen_structures = p.map_async(get_structure, gen_df['cif']).get()
    p.close()
    p.join()

    start_time = time()
    all_metrics = {}
    gen_metrics = GenMetrics(train_structures=train_structures,
                             test_structures=test_structures,
                             gen_structures=gen_structures,
                         )
    
    # all_metrics = gen_metrics.metrics
    # all_metrics['validity'] = gen_metrics.validity
    all_metrics['novelty'] = gen_metrics.novelty
    all_metrics['uniqueness'] = gen_metrics.uniqueness

    end_time = time()
    print('Time used: {:.2f} s'.format(end_time - start_time))
    print(all_metrics)
    with open(args.output_path + f'metrics_{args.label}.json', 'w') as f:
        json.dump(all_metrics, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train_path', default='/data/zdcao/crystal_gpt/dataset/mp_20/sg_225/train_sg_225.csv', help='')
    parser.add_argument('--test_path', default='/data/zdcao/crystal_gpt/dataset/mp_20/sg_225/test_sg_225.csv', help='')
    parser.add_argument('--gen_path', default='/data/zdcao/crystal_gpt/data/adam_bs_100_lr_0.0001_decay_0_clip_1_A_119_W_28_N_21_Nf_5_K_48_16_h0_256_l_4_H_8_k_16_m_32_drop_0.3/temp_1.0/output_225.csv', help='')
    parser.add_argument('--output_path', default='//data/zdcao/crystal_gpt/data/adam_bs_100_lr_0.0001_decay_0_clip_1_A_119_W_28_N_21_Nf_5_K_48_16_h0_256_l_4_H_8_k_16_m_32_drop_0.3/temp_1.0/', help='filepath of the metrics output file')
    parser.add_argument('--label', default='225', help='output file label')
    parser.add_argument('--num_io_process', type=int, default=40, help='number of process used in multiprocessing io')
    args = parser.parse_args()

    main(args)
