#!/bin/zsh
conda init
conda activate crystalformer
python ./main.py --folder ./data/ --train_path ./mp_20/train_layer_test.csv --valid_path ./mp_20/val_layer_test.csv --epochs 200
python ./main.py --optimizer none --test_path ./mp_20/test_layer_test.csv --restore_path ./data/adam_bs_100_lr_0.0001_decay_0_clip_1_A_119_W_28_N_21_a_1_w_1_l_1_Nf_5_Kx_16_Kl_4_h0_256_l_16_H_16_k_64_m_64_e_32_drop_0.5/epoch_000100.pkl --spacegroup 65 --num_samples 10  --batchsize 200 --temperature 1.0 --elements C