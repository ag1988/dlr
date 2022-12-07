i=0
# for TASK in "shift" "cumsum" "cummax" "sort" "reverse" 
# for TASK in "context_shift" "masked_select" "masked_select_fixed"
# for TASK in "shift" "masked_select" "masked_select_fixed"
# for TASK in "cummax" "reverse" "masked_select"
# for TASK in "shift" "reverse" 
# for TASK in "reverse" "masked_select" "context_shift"
# for TASK in "cummax" "reverse" "masked_select" "masked_select_fixed" "context_shift"
# for TASK in "sort"
# for TASK in "solve" "solve_fixed"
# for TASK in "shift" "masked_select_fixed" "solve_fixed" "reverse" "solve" "context_shift" "masked_select" "sort" 
for TASK in "context_shift"
do
    # CUDA_VISIBLE_DEVICES=$i python -m train model=dlr experiment=dss-sequence1d dataset.task=$TASK model.layer.version=''  model.layer.dt_min=0.00001 model.layer.d_state=4096  model.layer.dt_max=0.00001 model.layer.kernel_to_real=real dataset.L=4096 dataset.samples_per_epoch=16000 loader.batch_size=16 optimizer.lr=0.0001 model.layer.lr.Lambda=0.0001 model.layer.lr.W=0.0001 model.n_layers=6 &
    # sleep 4
    # i=$((i+1))
    # CUDA_VISIBLE_DEVICES=$i python -m train model=dlr experiment=dss-sequence1d dataset.task=$TASK model.layer.kernel_type=attn dataset.L=4096 dataset.samples_per_epoch=16000 loader.batch_size=16 optimizer.lr=0.001 model.layer.attn_ff=0 &
    # sleep 4
    # i=$((i+1))
    # CUDA_VISIBLE_DEVICES=$i python -m train model=dss experiment=dss-sequence1d dataset.task=$TASK model.layer.Lambda_init='lin' model.layer.dt_min=0.0001 model.layer.d_state=4096  model.layer.dt_max=0.01 model.layer.kernel_to_real=real dataset.L=4096 dataset.samples_per_epoch=16000 loader.batch_size=16 optimizer.lr=0.001 model.layer.lr.Lambda=0.001 model.layer.lr.W=0.001 model.layer.lr.log_dt=0.001 model.n_layers=1 &
    # sleep 4
    # i=$((i+1))
    CUDA_VISIBLE_DEVICES=$i python -m train model=dlr experiment=dss-sequence1d dataset.task=$TASK model.layer.kernel_type=attn dataset.L=512 dataset.samples_per_epoch=16000 loader.batch_size=64 optimizer.lr=0.001 model.n_layers=2 model.layer.attn_ff=16 &
    # sleep 4
    # i=$((i+1))
    # CUDA_VISIBLE_DEVICES=$i python -m train model=dlr experiment=dss-sequence1d dataset.task=$TASK model=dlr model.layer.version=''  model.layer.dt_min=0.00001 model.layer.d_state=4096  model.layer.dt_max=0.00001 model.layer.kernel_to_real=real dataset.L=512 dataset.samples_per_epoch=16000 loader.batch_size=64 optimizer.lr=0.00005 model.layer.lr.Lambda=0.00005 model.layer.lr.W=0.00005 model.n_layers=6 &
    sleep 4
    i=$((i+1))
done


# CUDA_VISIBLE_DEVICES=0 python -m train model=dlr experiment=dss-sequence1d dataset.task=masked_select model=dss model.layer.kernel_type=attn dataset.L=4096 dataset.samples_per_epoch=16000 loader.batch_size=16 optimizer.lr=0.001 wandb=null

# mips
# CUDA_VISIBLE_DEVICES=0 python -m train model=dlr experiment=dss-sequence1d dataset.task=mips model=dlr model.layer.version=''  model.layer.dt_min=0.00001 model.layer.d_state=4096  model.layer.dt_max=0.00001 model.layer.kernel_to_real=real dataset.L=4096 dataset.samples_per_epoch=64000 loader.batch_size=4 optimizer.lr=0.0001 model.layer.lr.Lambda=0.0001 model.layer.lr.W=0.0001 loader.num_workers=0 wandb=null

# CUDA_VISIBLE_DEVICES=0 python -m train model=dss experiment=dss-sequence1d dataset.task=mips model.layer.Lambda_init='lin' model.layer.dt_min=0.0001 model.layer.d_state=4096  model.layer.dt_max=0.01 model.layer.kernel_to_real=real dataset.L=4096 dataset.samples_per_epoch=64000 loader.batch_size=4 optimizer.lr=0.001 model.layer.lr.Lambda=0.001 model.layer.lr.W=0.001 model.layer.lr.log_dt=0.001 model.n_layers=1 loader.num_workers=0 wandb=null

# CUDA_VISIBLE_DEVICES=1 python -m train model=dlr experiment=dss-sequence1d dataset.task=mips model=dss model.layer.kernel_type=attn dataset.L=4096 dataset.samples_per_epoch=64000 loader.batch_size=4 optimizer.lr=0.0001 loader.num_workers=0 wandb=null

# i=2
# for L in 16384  65536  262144 
# for L in 1048576
# do
    # CUDA_VISIBLE_DEVICES=$i python -m train experiment=dss-sequence1d dataset.task=shift model=dlr model.layer.version='' model.layer.d_state=4096 model.layer.dt_min=0.00001 model.layer.dt_max=0.00001 model.layer.kernel_to_real=real dataset.L=$L dataset.samples_per_epoch=8000 loader.batch_size=4 optimizer.lr=0.00001 model.layer.lr.Lambda=0.00001 model.layer.lr.W=0.00001 model.d_model=32 &
    # CUDA_VISIBLE_DEVICES=$i python -m train experiment=dss-sequence1d model=dss dataset.task=shift model.layer.Lambda_init='lin' model.layer.dt_min=0.0001 model.layer.d_state=4096  model.layer.dt_max=0.01 model.layer.kernel_to_real=real dataset.L=$L dataset.samples_per_epoch=8000 loader.batch_size=4 optimizer.lr=0.001 model.layer.lr.Lambda=0.001 model.layer.lr.W=0.001 model.layer.lr.log_dt=0.001 model.n_layers=1 model.d_model=32 &
    # CUDA_VISIBLE_DEVICES=$i python -m train experiment=dss-sequence1d dataset.task=shift model=sgconv model.layer.d_state=4096 model.layer.alpha_min=1 model.layer.alpha_max=1 model.layer.l_max=$L dataset.L=$L dataset.samples_per_epoch=8000 loader.batch_size=4 optimizer.lr=0.00001 model.d_model=32 &
#     sleep 4
#     i=$((i+1))
# done

# CUDA_VISIBLE_DEVICES=6 python -m train experiment=dss-sequence1d dataset.task=shift model=dlr model.layer.version='' model.layer.d_state=4096 model.layer.dt_min=0.00001 model.layer.dt_max=0.00001 model.layer.kernel_to_real=prod dataset.L=1048576 dataset.samples_per_epoch=8000 loader.batch_size=4 optimizer.lr=0.00001 model.layer.lr.Lambda=0.00001 model.layer.lr.W=0.00001 model.d_model=16 wandb=null


# CUDA_VISIBLE_DEVICES=0 python -m train model=dlr experiment=dss-sequence1d dataset.task=cummax model=dlr model.layer.version=''  model.layer.dt_min=0.00001 model.layer.d_state=4096  model.layer.dt_max=0.00001 model.layer.kernel_to_real=real dataset.L=4096 dataset.samples_per_epoch=16000 loader.batch_size=16 optimizer.lr=0.0001 model.layer.lr.Lambda=0.0001 model.layer.lr.W=0.0001 model.n_layers=6 wandb=null

# CUDA_VISIBLE_DEVICES=0 python -m train model=dlr experiment=dss-sequence1d dataset.task=cummax model=dss model.layer.kernel_type=attn dataset.L=4096 dataset.samples_per_epoch=16000 loader.batch_size=16 optimizer.lr=0.0001 wandb=null
