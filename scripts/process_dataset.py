import os
import sys
import re
import glob
from pathlib import Path
import tyro
from tqdm import tqdm
import traceback
import gc
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__) + '/..')
from pipeline import Pipeline

def main(
    input_root: str = "/media/SipDrive31/TotalCaptureVideos/",
    output_root: str = "results/TotalCaptureVideos",
    calib_root: str = "data/total_capture_camera_calib",
    static_camera: bool = False,
    max_frames: int = -1, # Process all frames
):
    input_path = Path(input_root)
    output_path = Path(output_root)
    calib_path = Path(calib_root)

    if not input_path.exists():
        print(f"Error: Input path {input_path} does not exist.")
        return

    # Find all video files
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    video_files = []
    
    print(f"Searching for videos in {input_path}...")
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if Path(file).suffix.lower() in video_extensions:
                video_files.append(Path(root) / file)
    
    # Sort for deterministic order
    video_files.sort()

    print(f"Found {len(video_files)} videos.")
    
    if len(video_files) == 0:
        return

    for video_file in tqdm(video_files, desc="Processing videos"):
        # Determine relative path to preserve structure
        try:
            rel_path = video_file.relative_to(input_path).parent
        except ValueError:
            # Should not happen given os.walk
            rel_path = Path(".")
            
        video_name = video_file.stem
        
        # Create output directory
        # Structure: output_root / rel_path / video_name
        current_output_dir = output_path / rel_path / video_name
        
        if current_output_dir.exists() and (current_output_dir / "results.pkl").exists():
             print(f"Skipping {video_file}, output already exists at {current_output_dir}")
             continue
        
        # Extract camera ID
        # Assuming format like TC_S1_acting1_cam4.mp4
        match = re.search(r'cam(\d+)', video_name)
        if not match:
            print(f"Could not determine camera ID for {video_name}, skipping.")
            continue
        
        cam_id = match.group(1)
        calib_file = calib_path / f"{cam_id}.txt"
        
        if not calib_file.exists():
            print(f"Calibration file {calib_file} not found, skipping.")
            continue

        print(f"Processing {video_file} with camera {cam_id} using calibration {calib_file}...")
        
        # Initialize pipeline for each video to ensure clean state
        # This destroys the previous instance and reloads config
        try:
            pipeline = Pipeline(static_cam=static_camera)
            
            # Override config calibration file
            pipeline.cfg.calib = str(calib_file.absolute())
            
            pipeline.__call__(
                str(video_file), 
                str(current_output_dir), 
                static_cam=static_camera,
                save_only_essential=True,
                max_frame=max_frames if max_frames > 0 else None
            )
            
            # Clean up
            del pipeline
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing {video_file}: {e}")
            traceback.print_exc()
            # Try to clean up even on error
            if 'pipeline' in locals():
                del pipeline
            gc.collect()
            torch.cuda.empty_cache()

if __name__ == '__main__':
    tyro.cli(main)
