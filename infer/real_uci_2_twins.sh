{
export CUDA_VISIBLE_DEVICES=1
config_path="config/a2/twins/tar-c-t.json"
ckpt_path="/svl/data/two-phase-flow/yuegao/WAFT_models/waftv2-ckpts/twins/zero-shot.pth"
data_root="/svl/data/two-phase-flow/yuegao/real_data/data_parts/UCI_data_250923"

cam2_path="${data_root}/cam2_q1_1000"
results_postfix="_waftv2-twins-zero-shot"




python inference_video.py \
    --cfg $config_path \
    --ckpt $ckpt_path \
    --video $cam2_path \
    --output ${cam2_path}${results_postfix}


exit 0
}
