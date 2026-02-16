"""
Billet Capture Demo - Save depth and camera images with automatic labeling.

Keyboard Controls:
    S       - Save current camera and depth images (auto-increments image number)
    B       - Next billet (increments billet number, resets image number to 1)
    D       - Cycle to next camera device
    R       - Reset image number to 1
    Q       - Quit

Output files are saved to ./billet_captures/ with naming:
    billet1_camera_1.png
    billet1_depth_1.png   (colorized)
    billet1_depth_1.npy   (raw depth data)
"""

import cv2
import numpy as np
import os
from datetime import datetime
from config import ConfigModel
from utilities.image_processing import (
    apply_cmap,
    color_map_from_txt,
    normalize_array,
    trim_outliers,
)
from utilities.reconstruction import Reconstruction3D
from utilities.visualization import Visualize3D
from utilities.gelsightmini import GelSightMini
from utilities.logger import log_message


class BilletCapture:
    def __init__(self, config: ConfigModel, output_dir: str = "./billet_captures"):
        self.config = config
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Auto-detect starting point from existing files
        self.billet_num, self.image_num = self._find_resume_point()
        
        # Store current frame data for saving
        self.current_camera_frame = None
        self.current_depth_map = None
        self.current_depth_rgb = None
    
    def _find_resume_point(self):
        """Scan output directory to find where to resume from."""
        import re
        
        max_billet = 0
        max_image_per_billet = {}
        
        # Pattern: billet{N}_camera_{M}.png or billet{N}_depth_{M}.png
        pattern = re.compile(r'billet(\d+)_(?:camera|depth)_(\d+)\.(?:png|npy)')
        
        for filename in os.listdir(self.output_dir):
            match = pattern.match(filename)
            if match:
                billet = int(match.group(1))
                image = int(match.group(2))
                
                max_billet = max(max_billet, billet)
                
                if billet not in max_image_per_billet:
                    max_image_per_billet[billet] = 0
                max_image_per_billet[billet] = max(max_image_per_billet[billet], image)
        
        if max_billet == 0:
            # No existing files, start fresh
            log_message("No existing captures found. Starting at billet 1, image 1.")
            return 1, 1
        else:
            # Resume from last billet, next image
            next_image = max_image_per_billet.get(max_billet, 0) + 1
            log_message(f"Resuming from billet {max_billet}, image {next_image}")
            return max_billet, next_image
        
    def get_filename_base(self):
        """Generate base filename from current billet and image number."""
        return f"billet{self.billet_num}"
    
    def get_status_text(self, device_idx=None):
        """Get current status text for display."""
        device_str = f" | Device: {device_idx}" if device_idx is not None else ""
        return f"Billet: {self.billet_num} | Next Image: {self.image_num}{device_str} | [S]ave [B]++ [D]evice [R]eset [Q]uit"
    
    def save_images(self):
        """Save current camera and depth images, then auto-increment image number."""
        if self.current_camera_frame is None or self.current_depth_map is None:
            log_message("No frames to save yet.")
            return False
            
        base = self.get_filename_base()
        img_num = self.image_num
        
        # Save camera image (convert RGB to BGR for OpenCV)
        camera_path = os.path.join(self.output_dir, f"{base}_camera_{img_num}.png")
        camera_bgr = cv2.cvtColor(self.current_camera_frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(camera_path, camera_bgr)
        
        # Save colorized depth image (ensure uint8)
        depth_png_path = os.path.join(self.output_dir, f"{base}_depth_{img_num}.png")
        depth_rgb_uint8 = self.current_depth_rgb.astype(np.uint8)
        depth_bgr = cv2.cvtColor(depth_rgb_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(depth_png_path, depth_bgr)
        
        # Save raw depth data as numpy array (float values)
        depth_npy_path = os.path.join(self.output_dir, f"{base}_depth_{img_num}.npy")
        np.save(depth_npy_path, self.current_depth_map)
        
        log_message(f"Saved: {base}_camera_{img_num}.png, {base}_depth_{img_num}.png, {base}_depth_{img_num}.npy")
        
        # Auto-increment image number for next save
        self.image_num += 1
        return True
    
    def handle_key(self, key):
        """Handle keyboard input. Returns False if should quit."""
        if key == ord('q') or key == ord('Q'):
            return False
        elif key == ord('s') or key == ord('S'):
            self.save_images()
        elif key == ord('b') or key == ord('B'):
            self.billet_num += 1
            self.image_num = 1
            log_message(f"Now on Billet {self.billet_num}, image number reset to 1")
        elif key == ord('r') or key == ord('R'):
            self.image_num = 1
            log_message(f"Image number reset to 1")
        return True


def add_status_bar(image, text, bar_height=40):
    """Add a status bar at the bottom of the image."""
    h, w = image.shape[:2]
    # Create black bar
    bar = np.zeros((bar_height, w, 3), dtype=np.uint8)
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (bar_height + text_size[1]) // 2
    cv2.putText(bar, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    # Stack
    return np.vstack([image, bar])


def run_billet_capture(config: ConfigModel, output_dir: str = "./billet_captures", device_index: int = None):
    WINDOW_TITLE = "Billet Capture - GelSight Mini"
    
    capture = BilletCapture(config, output_dir)
    
    reconstruction = Reconstruction3D(
        image_width=config.camera_width,
        image_height=config.camera_height,
        use_gpu=config.use_gpu,
    )
    
    if reconstruction.load_nn(config.nn_model_path) is None:
        log_message("Failed to load model. Exiting.")
        return
    
    # Optional 3D visualizer
    visualizer3D = None
    if config.pointcloud_enabled:
        visualizer3D = Visualize3D(
            pointcloud_size_x=config.camera_width,
            pointcloud_size_y=config.camera_height,
            save_path="",
            window_width=int(config.pointcloud_window_scale * config.camera_width),
            window_height=int(config.pointcloud_window_scale * config.camera_height),
        )
    
    cmap = color_map_from_txt(
        path=config.cmap_txt_path, is_bgr=config.cmap_in_BGR_format
    )
    
    # Initialize camera
    cam_stream = GelSightMini(
        target_width=config.camera_width, target_height=config.camera_height
    )
    devices = cam_stream.get_device_list()
    device_indices = list(devices.keys())
    log_message(f"Available camera devices: {devices}")
    
    # Determine starting device
    if device_index is not None:
        current_device_idx = device_index
    else:
        current_device_idx = config.default_camera_index
    
    # Try to find a working device
    def try_device(idx):
        """Try to open a device and return True if successful."""
        try:
            cam_stream.select_device(idx)
            cam_stream.start()
            # Try to read a frame to verify it works
            for _ in range(10):  # Give it a few tries
                frame = cam_stream.update(dt=0)
                if frame is not None:
                    return True
            return False
        except Exception as e:
            log_message(f"Device {idx} failed: {e}")
            return False
    
    # Try the selected device first, then cycle through others if it fails
    working_device = None
    devices_to_try = [current_device_idx] + [d for d in device_indices if d != current_device_idx]
    
    for dev_idx in devices_to_try:
        log_message(f"Trying device {dev_idx}...")
        if try_device(dev_idx):
            working_device = dev_idx
            log_message(f"Device {dev_idx} is working!")
            break
        else:
            log_message(f"Device {dev_idx} failed to read frames.")
    
    if working_device is None:
        log_message("ERROR: No working camera device found. Exiting.")
        return
    
    current_device_idx = working_device
    
    def switch_device(new_idx):
        """Switch to a different device."""
        nonlocal current_device_idx
        if cam_stream.camera is not None:
            cam_stream.camera.release()
        cam_stream.select_device(new_idx)
        cam_stream.start()
        current_device_idx = new_idx
        # Reset depth zeroing
        reconstruction.depth_map_zero_counter = 0
        reconstruction.depth_map_zero = np.zeros((config.camera_height, config.camera_width))
        log_message(f"Switched to device {new_idx}. Re-zeroing depth...")
    
    log_message(f"Output directory: {os.path.abspath(output_dir)}")
    log_message("Controls: [S]ave [B]illet++ [D]evice cycle [R]eset image# [Q]uit")
    
    try:
        while True:
            frame = cam_stream.update(dt=0)
            if frame is None:
                continue
            
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Compute depth
            depth_map, contact_mask, grad_x, grad_y = reconstruction.get_depthmap(
                image=frame_rgb,
                markers_threshold=(config.marker_mask_min, config.marker_mask_max),
            )
            
            if visualizer3D:
                visualizer3D.update(depth_map, gradient_x=grad_x, gradient_y=grad_y)
            
            if np.isnan(depth_map).any():
                continue
            
            # Process depth for display
            depth_map_trimmed = trim_outliers(depth_map, 1, 99)
            depth_map_normalized = normalize_array(array=depth_map_trimmed, min_divider=10)
            depth_rgb = apply_cmap(data=depth_map_normalized, cmap=cmap)
            
            # Store current frames for saving
            capture.current_camera_frame = frame_rgb.copy()
            capture.current_depth_map = depth_map.copy()
            capture.current_depth_rgb = depth_rgb.copy()
            
            # Create side-by-side display (camera | depth)
            # Add labels
            camera_label = np.zeros((30, frame_rgb.shape[1], 3), dtype=np.uint8)
            cv2.putText(camera_label, f"Camera Feed ({int(cam_stream.fps)} FPS)", 
                       (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            depth_label = np.zeros((30, depth_rgb.shape[1], 3), dtype=np.uint8)
            cv2.putText(depth_label, "Depth Map", 
                       (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            camera_with_label = np.vstack([camera_label, frame_rgb])
            depth_with_label = np.vstack([depth_label, depth_rgb])
            
            # Add spacing between images
            spacer = np.zeros((camera_with_label.shape[0], 20, 3), dtype=np.uint8)
            
            combined = np.hstack([camera_with_label, spacer, depth_with_label])
            
            # Scale display
            scale = config.cv_image_stack_scale
            combined = cv2.resize(
                combined,
                (int(combined.shape[1] * scale), int(combined.shape[0] * scale)),
                interpolation=cv2.INTER_NEAREST,
            )
            
            # Add status bar
            display = add_status_bar(combined, capture.get_status_text(current_device_idx))
            
            # Convert to BGR for OpenCV display
            display_bgr = cv2.cvtColor(display.astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imshow(WINDOW_TITLE, display_bgr)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key != 255:  # Key was pressed
                if key == ord('d') or key == ord('D'):
                    # Cycle to next device
                    if len(device_indices) > 1:
                        current_pos = device_indices.index(current_device_idx) if current_device_idx in device_indices else -1
                        next_pos = (current_pos + 1) % len(device_indices)
                        next_device = device_indices[next_pos]
                        log_message(f"Switching to device {next_device}...")
                        switch_device(next_device)
                    else:
                        log_message("Only one device available.")
                elif not capture.handle_key(key):
                    break
            
            # Check if window was closed
            if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                for _ in range(5):
                    cv2.waitKey(1)
                break
                
    except KeyboardInterrupt:
        log_message("Exiting...")
    finally:
        if cam_stream.camera is not None:
            cam_stream.camera.release()
        cv2.destroyAllWindows()
        if visualizer3D:
            visualizer3D.visualizer.destroy_window()


if __name__ == "__main__":
    import argparse
    from config import GSConfig
    
    parser = argparse.ArgumentParser(
        description="Capture billet images with automatic labeling."
    )
    parser.add_argument(
        "--gs-config",
        type=str,
        default=None,
        help="Path to the JSON configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./billet_captures",
        help="Directory to save captured images (default: ./billet_captures)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Camera device index to use (default: auto-detect from config or first working device)",
    )
    
    args = parser.parse_args()
    
    if args.gs_config is None:
        args.gs_config = "default_config.json"
    
    gs_config = GSConfig(args.gs_config)
    run_billet_capture(config=gs_config.config, output_dir=args.output_dir, device_index=args.device)
