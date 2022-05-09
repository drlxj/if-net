bsub -n 10 -W 12:00 -R "rusage[mem=6000, ngpus_excl_p=2]" -R "select[gpu_model0==NVIDIAGeForceGTX1080Ti]" ./train.sh
