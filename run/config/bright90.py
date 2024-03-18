import subprocess 
import numpy as np 
import time 

dataset = 'mp'
nickname = 'mp-'+dataset + 'sort-xyz-embed' 

###############################
atom_types = 119

Nf = 5
Kx, Kl = 16, 4
h0_size = 256
transformer_layers = 4
num_heads = 8
key_size = 32
model_size = 64
embed_size = 32
dropout_rate = 0.3

optimizer = 'adam'
weight_decay = 0.0 
lr = 1e-4
lr_decay = 0.0
clip_grad = 1.0
batchsize = 100
epochs = 10000

lamb_a, lamb_w, lamb_l = 1.0, 1.0, 1.0

num_io_process = 40

if dataset == 'perov':
    n_max = 6
    wyck_types = 10

    train_path = '/home/wanglei/cdvae/data/perov_5/train.csv'
    valid_path = '/home/wanglei/cdvae/data/perov_5/val.csv'
    test_path = '/home/wanglei/cdvae/data/perov_5/test.csv'

elif dataset == 'mp':
    n_max = 21
    wyck_types = 28

    train_path = '/home/wanglei/cdvae/data/mp_20/train.csv'
    valid_path = '/home/wanglei/cdvae/data/mp_20/val.csv'
    test_path = '/home/wanglei/cdvae/data/mp_20/test.csv'

elif dataset == 'carbon':
    n_max = 25
    wyck_types = 28

    train_path = '/home/wanglei/cdvae/data/carbon_24/train.csv'
    valid_path = '/home/wanglei/cdvae/data/carbon_24/val.csv'
    test_path = '/home/wanglei/cdvae/data/carbon_24/test.csv'
else:
    print (dataset)


###############################
prog = '../src/main.py'
resfolder = '/data/wanglei/crystalgpt/' + nickname  + '/' 

def submitJob(bin,args,jobname,logname,run=False,wait=None):

    #prepare the job file 
    job='''#!/bin/bash -l
#SBATCH --partition=a800
#SBATCH --nodes=1
#SBATCH --cpus-per-task=%g
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=%s
#SBATCH --output=%s
#SBATCH --error=%s'''%(num_io_process,jobname,logname,logname)

    if wait is not None:
        dependency ='''
#SBATCH --dependency=afterany:%d\n'''%(wait)
        job += dependency 

    job += '''
#export XLA_PYTHON_CLIENT_PREALLOCATE=false
conda activate py310
echo "The current job ID is $SLURM_JOB_ID"
echo "Running on $SLURM_JOB_NUM_NODES nodes:"
echo $SLURM_JOB_NODELIST
echo "Using $SLURM_NTASKS_PER_NODE tasks per node"
echo "A total of $SLURM_NTASKS tasks is used"\n
echo Job started at `date`\n'''

    job +='python '+ str(bin) + ' '
    for key, val in args.items():
        if isinstance(val, bool):
            job += (" --%s" % key if val else "")
        else:
            job += " --%s %s" % (key, val)
    job += '''
echo Job finished at `date`\n'''

    #print job
    jobfile = open("jobfile", "w")
    jobfile.write("%s"%job)
    jobfile.close()

    #submit the job 
    if run:
        cmd = ['sbatch', 'jobfile']
        time.sleep(0.1)
    else:
        cmd = ['cat','jobfile']

    subprocess.check_call(cmd)
    return None

