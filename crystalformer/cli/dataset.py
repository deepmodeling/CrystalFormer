import os
import lmdb
import pickle
import numpy as np
from crystalformer.src.utils import GLXYZAW_from_file
import warnings
warnings.filterwarnings("ignore")


def csv_to_lmdb(csv_file, lmdb_file, args):
    if os.path.exists(lmdb_file):
        os.remove(lmdb_file)
        print(f"Removed existing {lmdb_file}")

    values = GLXYZAW_from_file(csv_file,
                               atom_types=args.atom_types,
                               wyck_types=args.wyck_types,
                               n_max=args.n_max,
                               num_workers=args.num_workers)
    keys = np.arange(len(values[0]))

    env = lmdb.open(
        lmdb_file,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(100e9),
    )

    with env.begin(write=True) as txn:
        for key, value in zip(keys, values):
            txn.put(str(key).encode("utf-8"), pickle.dumps(value))

    print(f"Successfully converted {csv_file} to {lmdb_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_max', type=int, default=21, help='The maximum number of atoms in the cell')
    parser.add_argument('--atom_types', type=int, default=119, help='Atom types including the padded atoms')
    parser.add_argument('--wyck_types', type=int, default=28, help='Number of possible multiplicites including 0')

    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=40)
    args = parser.parse_args()

    for i in ["test", "val", "train"]:
        csv_to_lmdb(
            os.path.join(args.path, f"{i}.csv"), 
            os.path.join(args.path, f"{i}.lmdb"),
            args
        )


if __name__ == "__main__":
    main()
