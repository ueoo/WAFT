{
export CUDA_VISIBLE_DEVICES=0
config_path="config/a2/dinov3/tar-c-t-sintel.json"
ckpt_path="/svl/data/two-phase-flow/yuegao/WAFT_models/waftv2-ckpts/dinov3/sintel.pth"
data_root="/svl/data/two-phase-flow/yuegao/boiling_water"

cam_path="${data_root}/render_simple_cam1"
results_postfix="_waftv2-dinov3-sintel"


python inference_video.py \
    --cfg $config_path \
    --ckpt $ckpt_path \
    --video $cam_path \
    --output ${cam_path}${results_postfix}


exit 0
}
