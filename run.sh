CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --master_port 45543 --nproc_per_node=1 test1.py --batch 1
