#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
dataset="nerf_synthetic"

prior_variance=0.01
posterior_lr=0.0001
AUSE_weight=5
KLweight=0.001
datasetargs="--resolution 2 --white_background"
GSargs="--eval --sh_degree 1"
bayes_params="--switch_to_beyesian 16000 --prior_xyz_variance $prior_variance --posterior_xyz_lr $posterior_lr --prior_features_variance $prior_variance --posterior_features_lr $posterior_lr --prior_opacity_variance $prior_variance --posterior_opacity_lr $posterior_lr --KL_weight $KLweight --AUSE_weight $AUSE_weight"

for scene in "chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship"
do
    input_path=${dataset}/${scene}
    output_path=output/BLENDER_${scene}
    python train.py -s $input_path --model_path $output_path $bayes_params $GSargs $datasetargs
    python render.py --model_path $output_path
    python compute_metrics.py --experiment_path ${output_path}/test/ours_30000
done
