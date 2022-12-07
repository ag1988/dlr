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

# OUT="outputs/out_dlr_pfsg512_7_kl16_bs16_n11"
# mkdir -p $OUT
# # COMMAND="CUDA_VISIBLE_DEVICES=0,1,2,3 python test_slurm.py"
# COMMAND="python -m train experiment=dss-pathfinder-segmentation-512 model.n_layers=12 model=dlr model.layer.version='' model.layer.dt_min=0.0001 model.layer.dt_max=0.1 model.layer.lr.Lambda=0.00001 model.layer.lr.W=0.00001 model.layer.d_state=2048 optimizer.lr=0.00001 loader.batch_size=2 model.layer.max_kernel_length=32768 trainer.gpus=7 trainer.find_unused_parameters=false trainer.save_val_outputs=false model.d_model=64"

OUT="outputs/out_dss_pfsg512_12_kl15_bs14_n11"
mkdir -p $OUT
# COMMAND="CUDA_VISIBLE_DEVICES=0,1,2,3 python test_slurm.py"
COMMAND="python -m train experiment=dss-pathfinder-segmentation-512 model.n_layers=12 model=dss model.layer.Lambda_init='lin' model.layer.dt_min=0.0001 model.layer.dt_max=0.01 model.layer.lr.Lambda=0.0001 model.layer.lr.W=0.0001 model.layer.d_state=2048 optimizer.lr=0.0001 model.layer.lr.log_dt=0.0001 loader.batch_size=2 model.layer.max_kernel_length=32768 trainer.gpus=7 trainer.find_unused_parameters=false trainer.save_val_outputs=false model.d_model=64"

# OUT="outputs/out_dlr-lost"
# mkdir -p $OUT
# COMMAND="python -m train experiment=dss-listops-subtrees model=dlr model.layer.version='' model.layer.dt_min=0.0001 model.layer.dt_max=0.1 model.layer.lr.Lambda=0.0008 model.layer.lr.W=0.0008  model.layer.d_state=1024 optimizer.lr=0.0008 loader.batch_size=32 dataset.l_min=7000  dataset.l_max=8192 trainer.save_val_outputs=false model.n_layers=18"

# export WANDB_API_KEY="b94322b308811f1d88bcc8a451609abd1d15bd3a"

sbatch << EOT
#!/bin/sh -l
#SBATCH --job-name=$OUT
#SBATCH --output=$OUT/std.out  # redirect stdout
#SBATCH --error=$OUT/std.err   # redirect stderr
#SBATCH --partition=gpu-joberant  # (see resources section)
#SBATCH --time=4300            # max time (minutes)
#SBATCH --signal=USR1@120      # how to end job when time’s up
#SBATCH --nodes=1              # number of machines
#SBATCH --ntasks=1             # number of processes
#SBATCH --mem=20000            # CPU memory (MB)
#SBATCH --cpus-per-task=4      # CPU cores per process
#SBATCH --gpus=7               # GPUs in total
#SBATCH --constraint="tesla_v100"

$COMMAND

EOT

# ssh ankitg@c-003.cs.tau.ac.il
# bash slurm.sh
# squeue --me            
# scancel <job-id>       
# detailed info about a job:   scontrol show jobid -dd <jobid>

#SBATCH --exclude=n-307        # n1,n2,..
#SBATCH --nodelist=n-301       # node to run on n1,n2,..


#SBATCH --constraint="geforce_rtx_3090"
# tesla_v100, quadro_rtx_8000, geforce_rtx_3090, titan_xp, geforce_rtx_2080,a100,a5000,a6000


# sbatch << EOT
# #!/bin/sh -l
# #SBATCH --job-name=$OUT
# #SBATCH --output=$OUT/std.out  # redirect stdout
# #SBATCH --error=$OUT/std.err   # redirect stderr
# #SBATCH --partition=killable   # (see resources section)
# #SBATCH --time=1400            # max time (minutes)
# #SBATCH --signal=USR1@120      # how to end job when time’s up
# #SBATCH --nodes=1              # number of machines
# #SBATCH --ntasks=1             # number of processes
# #SBATCH --mem=30000            # CPU memory (MB)
# #SBATCH --cpus-per-task=4      # CPU cores per process
# #SBATCH --gpus=1               # GPUs in total
# #SBATCH --constraint="a100"
