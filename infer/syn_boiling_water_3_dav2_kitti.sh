{
export CUDA_VISIBLE_DEVICES=1
config_path="config/a2/dav2/tar-c-t-kitti.json"
ckpt_path="/svl/data/two-phase-flow/yuegao/WAFT_models/waftv2-ckpts/dav2/kitti.pth"
data_root="/svl/data/two-phase-flow/yuegao/boiling_water"

cam_path="${data_root}/render_simple_cam3"
results_postfix="_waftv2-dav2-kitti"


python inference_video.py \
    --cfg $config_path \
    --ckpt $ckpt_path \
    --video $cam_path \
    --output ${cam_path}${results_postfix}


exit 0
}
