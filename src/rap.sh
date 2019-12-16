epoch=$1
gpu=$2
#python train.py -m HRPInception -d RAP -c 51 -e $epoch -s v1 -g $gpu
python test.py -m HRPInception -d RAP -c 51 -s thr3 -g $gpu -w v1_
#python train.py -m HRPInception -d RAP -c 51 -e $epoch -s v2 -g $gpu
python test.py -m HRPInception -d RAP -c 51 -s thr3 -g $gpu -w v2_