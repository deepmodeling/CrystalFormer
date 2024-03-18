#!/usr/bin/env python
import sys , os 

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action='store_true', help="Run or not")
    parser.add_argument("--waitfor", type=int, help="wait for this job for finish")
    input = parser.parse_args()


    #this import might overwrite the above default parameters 
    #########################################################
    import socket, getpass
    machinename = socket.gethostname()
    username = getpass.getuser()
    print ('\n', username, 'on', machinename, '\n')
    if 'ip-10-0-0-26' in machinename:
        from config.aws import * 
    elif 'ln01' in machinename:
        from config.ln01 import * 
    elif 'bright90' in machinename:
        from config.bright90 import * 
    else:
        print ('where am I ?', machinename)
        sys.exit(1)
    #########################################################

    from pygit2 import Repository
    head = Repository('.').head
    branch = head.shorthand + "-" + head.target.hex[:5]

    resfolder = os.path.join(resfolder, branch) + '/'

    jobdir='../jobs/' + nickname + '/'
    jobdir = os.path.join(jobdir, branch) + '/'

    cmd = ['mkdir', '-p', jobdir]
    subprocess.check_call(cmd)

    cmd = ['mkdir', '-p', resfolder]
    subprocess.check_call(cmd)
    
    if True:
                args = {'n_max':n_max, 
                        'atom_types': atom_types, 
                        'wyck_types': wyck_types, 
                        'folder':resfolder,
                        'Nf':Nf, 
                        'Kx':Kx, 
                        'Kl':Kl,
                        'h0_size': h0_size, 
                        'transformer_layers':transformer_layers, 
                        'num_heads':num_heads, 
                        'key_size':key_size,
                        'model_size':model_size, 
                        'embed_size':embed_size, 
                        'lr':lr, 
                        'lr_decay': lr_decay, 
                        'weight_decay': weight_decay, 
                        'clip_grad': clip_grad, 
                        'batchsize': batchsize,
                        'epochs': epochs, 
                        'optimizer': optimizer, 
                        'train_path' : train_path,  
                        'valid_path' : valid_path,  
                        'test_path' : test_path,  
                        'dropout_rate' : dropout_rate,  
                        'num_io_process' : num_io_process, 
                        'lamb_a': lamb_a,  
                        'lamb_w': lamb_w,  
                        'lamb_l': lamb_l,  
                        }

                logname = jobdir 
                for arg, value in args.items():
                    if isinstance(value, bool):
                        logname += ("%s_" % arg if value else "")
                    elif not ('_path' in arg or 'folder' in arg):
                        if '_' in arg:
                            arg = "".join([s[0] for s in arg.split('_')])
                        logname += "%s%s_" % (arg, value)
                logname = logname[:-1] + '.log'

                jobname = os.path.basename(os.path.dirname(logname))

                jobid = submitJob(prog,args,jobname,logname,run=input.run, wait=input.waitfor)
