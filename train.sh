python train.py -std_dev 0.1 0.01 -res 32 -m ShapeNet32Vox -batch_size 4 
#nvidia-smi -c 0 -g 0
python train.py -std_dev 0.1 0.01 -res 128 -m ShapeNet128Vox -batch_size 6
