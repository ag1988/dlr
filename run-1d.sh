# i=0
# for TASK in "cumsum" "cummax" "reverse" "sort" "context_shift"
# for TASK in "shift" "masked_select" "masked_select_fixed"
# for TASK in "cummax" "reverse" "masked_select"
# for TASK in "shift" "reverse" 
# do
#     CUDA_VISIBLE_DEVICES=$i python -m train model=dlr experiment=dss-sequence1d dataset.task=$TASK model=dlr model.layer.version=''  model.layer.dt_min=0.00001 model.layer.d_state=4096  model.layer.dt_max=0.00001 model.layer.kernel_to_real=real dataset.L=4096 dataset.samples_per_epoch=16000 loader.batch_size=16 optimizer.lr=0.00005 model.layer.lr.Lambda=0.00005 model.layer.lr.W=0.00005 model.n_layers=1 &
    # CUDA_VISIBLE_DEVICES=$i python -m train model=dlr experiment=dss-sequence1d dataset.task=$TASK model=dss model.layer.kernel_type=attn dataset.L=4096 dataset.samples_per_epoch=16000 loader.batch_size=16 optimizer.lr=0.001 &
    # sleep 8
    # i=$((i+1))
# done


# CUDA_VISIBLE_DEVICES=0 python -m train model=dlr experiment=dss-sequence1d dataset.task=masked_select model=dss model.layer.kernel_type=attn dataset.L=4096 dataset.samples_per_epoch=16000 loader.batch_size=16 optimizer.lr=0.0001 wamdb=null

# mips
# CUDA_VISIBLE_DEVICES=0 python -m train model=dlr experiment=dss-sequence1d dataset.task=mips model=dlr model.layer.version=''  model.layer.dt_min=0.00001 model.layer.d_state=4096  model.layer.dt_max=0.00001 model.layer.kernel_to_real=real dataset.L=4096 dataset.samples_per_epoch=64000 loader.batch_size=4 optimizer.lr=0.0001 model.layer.lr.Lambda=0.0001 model.layer.lr.W=0.0001 loader.num_workers=0 wandb=null

# CUDA_VISIBLE_DEVICES=1 python -m train model=dlr experiment=dss-sequence1d dataset.task=mips model=dss model.layer.kernel_type=attn dataset.L=4096 dataset.samples_per_epoch=64000 loader.batch_size=4 optimizer.lr=0.0001 loader.num_workers=0 wandb=null

# i=4
# for L in 16384  65536  262144 1048576
# do
#     CUDA_VISIBLE_DEVICES=$i python -m train experiment=dss-sequence1d dataset.task=shift model=dlr model.layer.version='' model.layer.d_state=4096 model.layer.dt_min=0.00001 model.layer.dt_max=0.00001 model.layer.kernel_to_real=prod dataset.L=$L dataset.samples_per_epoch=8000 loader.batch_size=4 optimizer.lr=0.00002 model.layer.lr.Lambda=0.00002 model.layer.lr.W=0.00002 model.d_model=32 &
#     sleep 8
#     i=$((i+1))
# done

# CUDA_VISIBLE_DEVICES=6 python -m train experiment=dss-sequence1d dataset.task=shift model=dlr model.layer.version='' model.layer.d_state=4096 model.layer.dt_min=0.00001 model.layer.dt_max=0.00001 model.layer.kernel_to_real=prod dataset.L=1048576 dataset.samples_per_epoch=8000 loader.batch_size=4 optimizer.lr=0.00001 model.layer.lr.Lambda=0.00001 model.layer.lr.W=0.00001 model.d_model=32 wandb=null


# CUDA_VISIBLE_DEVICES=0 python -m train model=dlr experiment=dss-sequence1d dataset.task=cummax model=dlr model.layer.version=''  model.layer.dt_min=0.00001 model.layer.d_state=4096  model.layer.dt_max=0.00001 model.layer.kernel_to_real=real dataset.L=4096 dataset.samples_per_epoch=16000 loader.batch_size=16 optimizer.lr=0.0001 model.layer.lr.Lambda=0.0001 model.layer.lr.W=0.0001 model.n_layers=6 wandb=null

# CUDA_VISIBLE_DEVICES=0 python -m train model=dlr experiment=dss-sequence1d dataset.task=cummax model=dss model.layer.kernel_type=attn dataset.L=4096 dataset.samples_per_epoch=16000 loader.batch_size=16 optimizer.lr=0.0001 wandb=null
