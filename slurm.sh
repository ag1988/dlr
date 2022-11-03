###! DO NOT use CUDA_VISIBLE_DEVICES ON SLURM!!!

# OUT="outputs/out_varcopy_dlr_real"
# mkdir -p $OUT
# COMMAND="python -m train model=dlr experiment=dss-copying-long model.layer.version=Lambda_imag_W_real model.layer.kernel_to_real=real dataset.inp_length=2048 dataset.l_memorize=32 model.n_layers=2 optimizer.lr=0.0001 model.layer.lr.Lambda=0.0001 model.layer.lr.W=0.0001"

# OUT="outputs/out_varcopy_dssexp_real"
# mkdir -p $OUT
# COMMAND="python -m train model=dss model.layer.version=exp experiment=dss-copying-long dataset.inp_length=2048 dataset.l_memorize=32 model.n_layers=2 optimizer.lr=0.0001"


# OUT="outputs/out_copy_dlr_prod"
# mkdir -p $OUT
# COMMAND="CUDA_VISIBLE_DEVICES=0 python -m train experiment=dss-sequence1d dataset.task=copy model=dlr model.layer.version=Lambda_imag_W_real model.layer.kernel_to_real=prod dataset.inp_length=16384 loader.batch_size=16 optimizer.lr=0.0001 model.layer.lr.Lambda=0.0001 model.layer.lr.W=0.0001"

# OUT="outputs/out_copy_dssexp_real"
# mkdir -p $OUT
# COMMAND="python -m train experiment=dss-sequence1d dataset.task=copy model=dss model.layer.version=exp dataset.inp_length=16384 optimizer.lr=0.0001"

# OUT="outputs/out_cumsum_dlr_prod"
# mkdir -p $OUT
# COMMAND="python -m train experiment=dss-sequence1d dataset.task=cumsum model=dlr model.layer.version=Lambda_imag_W_real model.layer.kernel_to_real=prod dataset.inp_length=16384 loader.batch_size=16 optimizer.lr=0.0001 model.layer.lr.Lambda=0.0001 model.layer.lr.W=0.0001"

# OUT="outputs/out_cumsum_dssexp_real"
# mkdir -p $OUT
# COMMAND="python -m train experiment=dss-sequence1d dataset.task=cumsum model=dss model.layer.version=exp dataset.inp_length=16384 loader.batch_size=16 optimizer.lr=0.0001"


# OUT="outputs/out_rev_dlr_prod"
# mkdir -p $OUT
# COMMAND="python -m train experiment=dss-sequence1d dataset.task=reverse model=dlr model.layer.version=Lambda_imag_W_real model.layer.kernel_to_real=prod dataset.inp_length=2048 dataset.samples_per_epoch=128000 optimizer.lr=0.0001 model.layer.lr.Lambda=0.0001 model.layer.lr.W=0.0001"
# COMMAND="python test_slurm.py"

# OUT="outputs/out_rev_dssexp_real"
# mkdir -p $OUT
# COMMAND="python -m train experiment=dss-sequence1d dataset.task=reverse model=dss model.layer.version=exp dataset.inp_length=2048 dataset.samples_per_epoch=128000 optimizer.lr=0.0001"


export WANDB_API_KEY="b94322b308811f1d88bcc8a451609abd1d15bd3a"

sbatch << EOT
#!/bin/sh -l
#SBATCH --job-name=$OUT
#SBATCH --output=$OUT/std.out  # redirect stdout
#SBATCH --error=$OUT/std.err   # redirect stderr
#SBATCH --partition=killable   # (see resources section)
#SBATCH --time=720             # max time (minutes)
#SBATCH --signal=USR1@120      # how to end job when time’s up
#SBATCH --nodes=1              # number of machines
#SBATCH --ntasks=1             # number of processes
#SBATCH --mem=20000            # CPU memory (MB)
#SBATCH --cpus-per-task=4      # CPU cores per process
#SBATCH --gpus=1               # GPUs in total
#SBATCH --constraint="geforce_rtx_3090"

$COMMAND

EOT

# ssh ankitg@c-003.cs.tau.ac.il
# bash slurm.sh
# squeue --me            
# scancel <job-id>       
# detailed info about a job:   scontrol show jobid -dd <jobid>

#SBATCH --exclude=n-307        # n1,n2,..
#SBATCH --nodelist=n-301       # node to run on n1,n2,..


