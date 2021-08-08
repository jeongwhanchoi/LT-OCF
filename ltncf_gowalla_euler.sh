cd code && python main.py --dataset="gowalla" --model="ltocf" --solver="euler" --adjoint=False --K=4 --learnable_time=True --dual_res=False --lr=1e-4 --lr_time=1e-6 --decay=1e-4 --topks="[20]" --load=1 --pretrain=1 --comment="learnable_time" --pretrained_file="gowalla-euler.pth.tar" --tensorboard=1 --gpuid=0