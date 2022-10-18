
# conda init pytorch
# CUDA_VISLBLE_DEVICES=0,1 python -m torch.distributed.launch  --nproc_per_node=2 train.py
# torchrun train.py
torchrun --nnodes=1 --standalone --nproc_per_node=2 train.py