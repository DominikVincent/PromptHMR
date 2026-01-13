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
import multiprocessing as mp

# Add project root to path
# We keep this at global scope so the spawned process can find the module
sys.path.insert(0, os.path.dirname(__file__) + '/..')
from pipeline import Pipeline

def process_single_video(
    video_file: Path, 
    current_output_dir: Path, 
    calib_file: Path, 
    static_camera: bool, 
    max_frames: int
):
    """
    This function runs in a completely separate operating system process.
    If it runs out of RAM and gets killed, it won't crash the main script.
    """
    try:
        print(f"\n[Worker] Starting process for: {video_file.name}")
        
        # Initialize pipeline inside the worker process
        # This ensures it gets a fresh block of memory
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
        
        # Explicit cleanup (though OS will reclaim memory when process ends)
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[Worker] Finished successfully: {video_file.name}")
        
    except Exception as e:
        print(f"[Worker] Python Error processing {video_file.name}: {e}")
        traceback.print_exc()
        # Exit with a non-zero code so the parent knows it failed
        sys.exit(1)

def main(
    input_root: str = "/media/SipDrive31/TotalCaptureVideos/",
    output_root: str = "results/TotalCaptureVideos",
    calib_root: str = "data/total_capture_camera_calib",
    static_camera: bool = False,
    max_frames: int = -1, # Process all frames
):
    # CRITICAL: Use 'spawn' for PyTorch/CUDA compatibility
    # 'fork' (default on Linux) can cause deadlocks with CUDA
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

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

    # Using tqdm for progress bar
    pbar = tqdm(video_files, desc="Processing videos")
    
    for video_file in pbar:
        # Determine relative path to preserve structure
        try:
            rel_path = video_file.relative_to(input_path).parent
        except ValueError:
            rel_path = Path(".")
            
        video_name = video_file.stem
        
        # Create output directory
        current_output_dir = output_path / rel_path / video_name
        failed_marker = current_output_dir / "FAILED_RUN.txt"
        
        if current_output_dir.exists():
            if (current_output_dir / "results.pkl").exists():
                pbar.write(f"Skipping {video_file.name}, output already exists.")
                continue
            if failed_marker.exists():
                pbar.write(f"Skipping {video_file.name}, previous run failed (marker found).")
                continue
        
        # Extract camera ID
        match = re.search(r'cam(\d+)', video_name)
        if not match:
            pbar.write(f"Could not determine camera ID for {video_name}, skipping.")
            continue
        
        cam_id = match.group(1)
        calib_file = calib_path / f"{cam_id}.txt"
        
        if not calib_file.exists():
            pbar.write(f"Calibration file {calib_file} not found, skipping.")
            continue

        pbar.write(f"Spawning worker for {video_file.name} (Cam {cam_id})...")
        
        # Create directory and failure marker before starting
        current_output_dir.mkdir(parents=True, exist_ok=True)
        with open(failed_marker, "w") as f:
            f.write(f"Processing started for {video_file.name}\n")

        # --- MULTIPROCESSING EXECUTION ---
        
        # Create the child process
        p = mp.Process(
            target=process_single_video,
            args=(video_file, current_output_dir, calib_file, static_camera, max_frames)
        )
        
        # Start the process
        p.start()
        
        # Wait for the process to finish
        p.join()
        
        # Check how it finished
        if p.exitcode == 0:
            # Success
            if failed_marker.exists():
                failed_marker.unlink()
            pass 
        elif p.exitcode == 247 or p.exitcode == -9:
            # -9 is SIGKILL, 247 is the python representation of it
            pbar.write(f"⚠️  CRASH DETECTED: {video_file.name} was killed by the OS (likely Out of RAM).")
            pbar.write(f"⚠️  Skipping this video and continuing to the next...")
            with open(failed_marker, "a") as f:
                f.write(f"Crashed with exit code {p.exitcode} (likely OOM)\n")
        else:
            pbar.write(f"❌ Worker failed with exit code {p.exitcode} for {video_file.name}")
            with open(failed_marker, "a") as f:
                f.write(f"Failed with exit code {p.exitcode}\n")

if __name__ == '__main__':
    tyro.cli(main)