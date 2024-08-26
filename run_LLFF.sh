#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
dataset="nerf_llff_data"

# # First remove camera distortion from the images
# mkdir -p ${dataset}/undistorted
# for scene in "fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex"
# do
#     echo "********************************"
#     echo "Processing scene: ${scene}"
#     echo "********************************"
#     imagepath="${dataset}/${scene}/images"
#     inputpath="${dataset}/${scene}/sparse/0"
#     outputpath="${dataset}/undistorted/${scene}"
#     colmap image_undistorter --image_path ${imagepath} --input_path ${inputpath} --output_path ${outputpath} --output_type COLMAP
#     cd ${outputpath}/sparse
#     mkdir 0
#     mv *.bin 0
# done


# Then run the training
prior_variance=0.01
posterior_lr=0.0001
AUSE_weight=5
KLweight=0.001
datasetargs="--resolution 8 --opacity_reset_interval 10000000 --percent_dense 0.00001"
GSargs="--eval --sh_degree 1"
bayes_params="--switch_to_beyesian 16000 --prior_xyz_variance $prior_variance --posterior_xyz_lr $posterior_lr --prior_features_variance $prior_variance --posterior_features_lr $posterior_lr --prior_opacity_variance $prior_variance --posterior_opacity_lr $posterior_lr --KL_weight $KLweight --AUSE_weight $AUSE_weight"

for scene in "fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex"
do
    input_path=${dataset}/undistorted/${scene}
    output_path=output/LLFF_${scene}
    python train.py -s $input_path --model_path $output_path $bayes_params $GSargs $datasetargs
    python render.py --model_path $output_path
    python compute_metrics.py --experiment_path ${output_path}/test/ours_30000
done
