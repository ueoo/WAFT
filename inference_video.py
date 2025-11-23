import argparse
import glob
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm, trange

from config.parser import parse_args
from inference_tools import InferenceWrapper
from model import fetch_model
from utils.flow_viz import flow_to_image
from utils.utils import load_ckpt


@torch.no_grad()
def inference_video(model, args, frames_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Setup input source
    is_video_file = os.path.isfile(frames_path)

    image_files = sorted(
        [
            os.path.join(frames_path, f)
            for f in os.listdir(frames_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]
    )
    total_frames = len(image_files)

    print(f"Processing frames from {frames_path}...")

    # Prepare first frame
    prev_frame_bgr = cv2.imread(image_files[0])
    prev_frame = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2RGB)
    prev_frame = torch.from_numpy(prev_frame).permute(2, 0, 1).float()[None].cuda()

    for frame_idx in trange(1, total_frames):
        curr_frame_bgr = cv2.imread(image_files[frame_idx])
        if curr_frame_bgr is None:
            print(f"Failed to read {image_files[frame_idx]}")
            continue

        cur_frame_name = image_files[frame_idx].split("/")[-1].split(".")[0]

        curr_frame = cv2.cvtColor(curr_frame_bgr, cv2.COLOR_BGR2RGB)
        curr_frame_tensor = torch.from_numpy(curr_frame).permute(2, 0, 1).float()[None].cuda()

        # Inference
        output = model.calc_flow(prev_frame, curr_frame_tensor)

        # Get the final flow (usually the last one in the list)
        if "flow" in output and len(output["flow"]) > 0:
            flow = output["flow"][-1]

            # Visualize
            flow_numpy = flow[0].permute(1, 2, 0).cpu().numpy()
            flow_vis = flow_to_image(flow_numpy, convert_to_bgr=True)

            cv2.imwrite(os.path.join(output_path, f"{cur_frame_name}.jpg"), flow_vis)
            np.save(os.path.join(output_path, f"{cur_frame_name}.npy"), flow_numpy)

        # Update previous frame
        prev_frame = curr_frame_tensor

    print(f"Done. Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", help="experiment configure file name", required=True, type=str)
    parser.add_argument("--ckpt", help="checkpoint path", required=True, type=str)
    parser.add_argument("--video", help="path to video file or directory of frames", required=True, type=str)
    parser.add_argument("--output", help="output directory", default="demo_output", type=str)
    parser.add_argument("--scale", help="scale factor for input images", default=0.0, type=float)

    # Parse args using the project's parser utility
    args = parse_args(parser)

    # Load model
    model = fetch_model(args)
    load_ckpt(model, args.ckpt)
    model = model.cuda()
    model.eval()

    # Inference wrapper
    wrapped_model = InferenceWrapper(
        model, scale=args.scale, train_size=args.image_size, pad_to_train_size=False, tiling=False
    )

    inference_video(wrapped_model, args, args.video, args.output)


if __name__ == "__main__":
    main()
