epoch=$1
gpu=$2
python train.py -m Inception -d PETA -c 61 -e $epoch -s v1 -g $gpu -w ../models/PETA/Inception/v1_epoch090_valloss0.985493.hdf5
#python test.py -m HRPInception -d PETA -c 61 -s thr3 -g $gpu -w v1_
#python train.py -m HRPInception -d PETA -c 61 -e $epoch -s v2 -g $gpu
#python test.py -m HRPInception -d PETA -c 61 -s thr3 -g $gpu -w v2_