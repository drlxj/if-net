python evaluate.py -reconst -generation_path experiments/iVoxels_dist-0.5_0.5_sigmas-0.1_0.01_v32_mShapeNet32Vox/evaluation_10_@256/generation/
python evaluate.py -voxels -res 32
python evaluate_gather.py -voxel_input -res 32 -generation_path experiments/iVoxels_dist-0.5_0.5_sigmas-0.1_0.01_v32_mShapeNet32Vox/evaluation_10_@256/generation/
