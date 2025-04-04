# alexandria dataset: https://alexandria.icams.rub.de/, https://archive.materialscloud.org/record/2022.126
# This script is used to process the raw data from alexandria dataset, and save the data into csv files
import os
import json
import bz2
import pandas as pd
import multiprocessing as mp
from sklearn.model_selection import train_test_split


def get_file_name(filepath):
    filename_list = []
    for root, _, files in os.walk(filepath):
        for file in files:
            if file.endswith(".json.bz2"):
                filename_list.append(os.path.join(root, file))
    return filename_list


def get_data_from_file(filename):
    with bz2.open(filename) as fh:
        data = json.loads(fh.read().decode('utf-8'))
        fh.close()

    print(len(data["entries"]))
    entries = data["entries"]
    # save this key information to a dataframe
    df = pd.DataFrame([{'e_above_hull': entry['data']['e_above_hull'],
                        'e_form': entry['data']['e_form'],
                        'mat_id': entry['data']['mat_id'],
                        'formula': entry['data']['formula'],
                        'elements': entry['data']['elements'],
                        'spg': entry['data']['spg'],
                        'band_gap_dir': entry['data']['band_gap_dir'],
                        'band_gap_ind': entry['data']['band_gap_ind'],
                        'nsites': entry['data']['nsites'],
                        'structure': entry['structure']} for entry in entries])
    # screening the data
    df = df[(df['e_above_hull'] <= 0.1) & (df['nsites'] <= 20)]

    return df


def main(args):
    filename_list = get_file_name(args.input_path)
    print(len(filename_list))
    with mp.Pool(args.num_io_process) as pool:
        df_list = pool.map_async(get_data_from_file, filename_list).get()
    df_total = pd.concat(df_list, axis=0)

    print("total data: ", df_total.shape)
    if args.ratio < 1.0:
        df_total = df_total.sample(frac=args.ratio, random_state=42)
        print("random sampled data: ", df_total.shape)
    
    ########### split the data into train, val, test ###########
    train_data, val_test_data = train_test_split(df_total, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(val_test_data, test_size=0.5, random_state=42)

    print("train data: ",train_data.shape)
    print("val data: ", val_data.shape)
    print("test data: ", test_data.shape)
    print(f"will output the data to {args.output_path}")
    train_data.to_csv(f"{args.output_path}/train.csv", index=False)
    val_data.to_csv(f"{args.output_path}/val.csv", index=False)
    test_data.to_csv(f"{args.output_path}/test.csv", index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process ALEX data")
    parser.add_argument("--input_path", type=str, default="/data/zdcao/crystal_gpt/dataset/alex/origin/", help="path to the input data")
    parser.add_argument("--output_path", type=str, default="/data/zdcao/crystal_gpt/dataset/alex/alex20_811/", help="path to the output data")
    parser.add_argument("--ratio", type=float, default=1.0, help="ratio of the data to be used")
    parser.add_argument('--num_io_process', type=int, default=20, help='number of process used in multiprocessing io')

    args = parser.parse_args()
    main(args)
