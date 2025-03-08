#!/usr/bin/env python3
"""
Three-Camera SuperGlue Feature Matching for 3D Reconstruction

This script processes images from three cameras, extracting and matching features
using SuperPoint and SuperGlue. It integrates camera calibration data to improve 
matching accuracy and prepares the output for the depth estimation stage.

Original SuperGlue implementation by Paul-Edouard Sarlin, Daniel DeTone, and Tomasz Malisiewicz
Modified for a three-camera setup for 3D mapping pipeline.
"""

import argparse
import cv2
import json
import matplotlib.cm as cm
import numpy as np
import os
import torch
from pathlib import Path

# Import SuperGlue components (assumes same structure as original script)
from models.matching import Matching
from models.utils import (AverageTimer, make_matching_plot_fast, frame2tensor)

torch.set_grad_enabled(False)
def load_tiff_rg8(tiff_path):
    """
    Load a TIFF RG8 file and convert it to grayscale
    
    Args:
        tiff_path: Path to the TIFF RG8 file
        
    Returns:
        Grayscale image as numpy array
    """
    # Load the TIFF file
    tiff_img = cv2.imread(str(tiff_path), cv2.IMREAD_UNCHANGED)
    
    # Check if image was loaded properly
    if tiff_img is None:
        print(f"Failed to load {tiff_path}")
        return None
    
    # Print info about the image
    print(f"TIFF image shape: {tiff_img.shape}, dtype: {tiff_img.dtype}")
    
    # Convert Bayer RG8 to grayscale
    # If it's a raw Bayer pattern, we can just take the values directly as grayscale
    # since SuperPoint works with grayscale anyway
    if len(tiff_img.shape) == 2:
        # It's already a 2D array, just use as is
        gray_img = tiff_img
    elif len(tiff_img.shape) == 3:
        # If it's already been debayered to RGB, convert to grayscale
        gray_img = cv2.cvtColor(tiff_img, cv2.COLOR_RGB2GRAY)
    else:
        print(f"Unexpected image format: {tiff_img.shape}")
        return None
    
    # Normalize to 8-bit if needed
    if gray_img.dtype == np.uint16:
        gray_img = (gray_img / 256).astype(np.uint8)
    
    return gray_img

class TrinocularMatcher:
    """Class to handle feature matching across three cameras"""
    
    def __init__(self, config, device='cuda'):
        """Initialize with configuration for SuperPoint and SuperGlue"""
        self.config = config
        self.device = device
        self.matching = Matching(config).eval().to(device)
        self.keys = ['keypoints', 'scores', 'descriptors']
        
    def load_calibration(self, calib_file):
        """Load camera calibration data"""
        try:
            with open(calib_file, 'r') as f:
                self.calib_data = json.load(f)
            
            # Extract reference camera and camera serials
            self.reference_camera = self.calib_data.get("reference_camera")
            self.cameras = self.calib_data.get("cameras", [])
            
            # Store individual camera calibrations
            self.camera_calibrations = {}
            for camera in self.cameras:
                if "individual_calibrations" in self.calib_data and camera in self.calib_data["individual_calibrations"]:
                    self.camera_calibrations[camera] = self.calib_data["individual_calibrations"][camera]
            
            # Store pairwise calibrations with proper transforms
            self.pairwise_calibrations = self.calib_data.get("pairwise_calibrations", {})
            
            print(f"Loaded calibration data from {calib_file}")
            print(f"Reference camera: {self.reference_camera}")
            print(f"Cameras: {self.cameras}")
            print(f"Pairwise calibrations: {list(self.pairwise_calibrations.keys())}")
            return True
        except Exception as e:
            print(f"Failed to load calibration: {e}")
            self.calib_data = None
            self.camera_calibrations = {}
            self.pairwise_calibrations = {}
            return False
            
    def undistort_image(self, image, camera_serial):
        """Undistort image using camera calibration parameters"""
        if not hasattr(self, 'camera_calibrations') or not self.camera_calibrations:
            return image
            
        try:
            # Get calibration for this camera
            if camera_serial not in self.camera_calibrations:
                print(f"No calibration found for camera {camera_serial}")
                return image
                
            calib = self.camera_calibrations[camera_serial]
                
            # Get camera matrix and distortion coefficients
            K = np.array(calib["camera_matrix"])
            dist = np.array(calib["distortion_coefficients"][0])  # Note: dist coeffs are in nested array
            
            # Create optimal new camera matrix
            h, w = image.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
            
            # Undistort
            undistorted = cv2.undistort(image, K, dist, None, newcameramtx)
            
            # Crop the image (optional)
            # x, y, w, h = roi
            # undistorted = undistorted[y:y+h, x:x+w]
            
            return undistorted
            
        except Exception as e:
            print(f"Undistortion failed for camera {camera_serial}: {e}")
            return image
    
    def extract_features(self, image_tensor):
        """Extract SuperPoint features from an image tensor"""
        return self.matching.superpoint({'image': image_tensor})
    
    def match_features(self, feats0, feats1, image0_tensor, image1_tensor):
        """Match features between two images using SuperGlue"""
        # Prepare data for matching
        data = {
            'image0': image0_tensor, 
            'image1': image1_tensor
        }
        
        # Add features data
        for k in self.keys:
            data[f'{k}0'] = feats0[k]
            data[f'{k}1'] = feats1[k]
            
        # Perform matching
        return self.matching(data)
    def auto_epipolar_threshold(self, kpts0, kpts1, matches, camera0, camera1, min_ratio=0.3):
        """Find minimum threshold that keeps at least min_ratio of matches"""
        valid = matches > -1
        if np.sum(valid) == 0:
            return 10.0  # Default if no matches
            
        # Get matched points
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        
        # Get fundamental matrix (reuse code from filter_matches_with_calibration)
        pair_key = f"{camera0}_{camera1}"
        if pair_key not in self.pairwise_calibrations:
            pair_key = f"{camera1}_{camera0}"  # Try reverse order
            is_reversed = True
        else:
            is_reversed = False
            
        if pair_key not in self.pairwise_calibrations:
            return 10.0  # Default if no calibration
            
        calib = self.pairwise_calibrations[pair_key]
        
        # Get the fundamental matrix
        if "fundamental_matrix" not in calib:
            return 10.0
        
        F = np.array(calib["fundamental_matrix"])
        
        # If the cameras are in reverse order, transpose F
        if is_reversed:
            F = F.T
        
        # Convert to homogeneous coordinates
        mkpts0_hom = np.hstack((mkpts0, np.ones((mkpts0.shape[0], 1))))
        mkpts1_hom = np.hstack((mkpts1, np.ones((mkpts1.shape[0], 1))))
        
        # Calculate epipolar lines
        epipolar_lines = (F @ mkpts0_hom.T).T
        
        # Calculate distances
        dists = np.abs(np.sum(mkpts1_hom * epipolar_lines, axis=1)) / np.sqrt(np.sum(epipolar_lines[:, :2]**2, axis=1))
        
        # Sort distances and find threshold that keeps min_ratio of points
        sorted_dists = np.sort(dists)
        idx = min(int(len(sorted_dists) * min_ratio), len(sorted_dists)-1)
        threshold = max(sorted_dists[idx], 2.0)  # At least 2.0 pixels
        
        return threshold
    def filter_matches_with_calibration(self, kpts0, kpts1, matches, confidence, camera0, camera1):
        """Filter matches using epipolar geometry from calibration"""
        if not hasattr(self, 'pairwise_calibrations') or not self.pairwise_calibrations:
            return matches, confidence
            
        try:
            # Get the fundamental matrix for this camera pair
            pair_key = f"{camera0}_{camera1}"
            if pair_key not in self.pairwise_calibrations:
                pair_key = f"{camera1}_{camera0}"  # Try reverse order
                is_reversed = True
            else:
                is_reversed = False
                
            if pair_key not in self.pairwise_calibrations:
                print(f"No pairwise calibration found for {camera0}-{camera1}")
                return matches, confidence
                
            calib = self.pairwise_calibrations[pair_key]
            
            # Get the fundamental matrix
            if "fundamental_matrix" in calib:
                F = np.array(calib["fundamental_matrix"])
                
                # If the cameras are in reverse order, transpose F
                if is_reversed:
                    F = F.T
            else:
                print(f"No fundamental matrix found for {pair_key}")
                return matches, confidence
                
            # Apply epipolar constraint
            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            
            # Convert to homogeneous coordinates
            pts0_hom = np.hstack((mkpts0, np.ones((mkpts0.shape[0], 1))))
            pts1_hom = np.hstack((mkpts1, np.ones((mkpts1.shape[0], 1))))
            
            # Calculate epipolar lines
            lines = (F @ pts0_hom.T).T
            
            # Calculate distances from points to their corresponding epipolar lines
            # d = (ax + by + c) / sqrt(a^2 + b^2)
            dists = np.abs(np.sum(pts1_hom * lines, axis=1)) / np.sqrt(np.sum(lines[:, :2]**2, axis=1))
            
            # Filter based on distance threshold (adjust if needed)
            threshold = self.auto_epipolar_threshold(kpts0, kpts1, matches, camera0, camera1)
            print(f"Auto epipolar threshold for {camera0}-{camera1}: {threshold:.2f} pixels")
            epi_mask = dists < threshold      

            # Update matches and confidence
            new_valid = np.zeros_like(matches, dtype=bool)
            new_valid[valid] = epi_mask
            filtered_matches = np.copy(matches)
            filtered_matches[valid][~epi_mask] = -1
            
            # Also filter confidence
            filtered_confidence = np.copy(confidence)
            filtered_confidence[valid][~epi_mask] = 0
            
            print(f"Epipolar filtering {camera0}-{camera1}: {np.sum(valid)} -> {np.sum(new_valid)} matches")
            
            return filtered_matches, filtered_confidence
            
        except Exception as e:
            print(f"Calibration-based filtering failed for {camera0}-{camera1}: {e}")
            return matches, confidence
            
    def process_image_triplet(self, image1, image2, image3, camera1_serial, camera2_serial, camera3_serial):
        """Process a triplet of images from three cameras"""
        timer = AverageTimer()
        
        # Undistort images if calibration is available
        image1_undist = self.undistort_image(image1, camera1_serial)
        image2_undist = self.undistort_image(image2, camera2_serial)
        image3_undist = self.undistort_image(image3, camera3_serial)
        timer.update('undistort')
        
        # Convert to tensors
        tensor1 = frame2tensor(image1_undist, self.device)
        tensor2 = frame2tensor(image2_undist, self.device)
        tensor3 = frame2tensor(image3_undist, self.device)
        timer.update('tensor')
        
        # Extract features
        feats1 = self.extract_features(tensor1)
        feats2 = self.extract_features(tensor2)
        feats3 = self.extract_features(tensor3)
        timer.update('extract')
        
        # Match features between camera pairs
        pred_1_2 = self.match_features(feats1, feats2, tensor1, tensor2)
        pred_1_3 = self.match_features(feats1, feats3, tensor1, tensor3)
        pred_2_3 = self.match_features(feats2, feats3, tensor2, tensor3)
        timer.update('match')
        
        # Extract matching results
        results = {
            'camera1_serial': camera1_serial,
            'camera2_serial': camera2_serial,
            'camera3_serial': camera3_serial,
            'pairs': {}
        }
        
        # Process matches for each pair
        # Pair 1-2
        kpts1 = feats1['keypoints'][0].cpu().numpy()
        kpts2 = feats2['keypoints'][0].cpu().numpy()
        matches_1_2 = pred_1_2['matches0'][0].cpu().numpy()
        confidence_1_2 = pred_1_2['matching_scores0'][0].cpu().numpy()
        
        # Apply epipolar filtering if calibration is available
        matches_1_2, confidence_1_2 = self.filter_matches_with_calibration(
            kpts1, kpts2, matches_1_2, confidence_1_2, camera1_serial, camera2_serial)
        
        # Pair 1-3
        kpts3 = feats3['keypoints'][0].cpu().numpy()
        matches_1_3 = pred_1_3['matches0'][0].cpu().numpy()
        confidence_1_3 = pred_1_3['matching_scores0'][0].cpu().numpy()
        
        matches_1_3, confidence_1_3 = self.filter_matches_with_calibration(
            kpts1, kpts3, matches_1_3, confidence_1_3, camera1_serial, camera3_serial)
        
        # Pair 2-3
        matches_2_3 = pred_2_3['matches0'][0].cpu().numpy()
        confidence_2_3 = pred_2_3['matching_scores0'][0].cpu().numpy()
        
        matches_2_3, confidence_2_3 = self.filter_matches_with_calibration(
            kpts2, kpts3, matches_2_3, confidence_2_3, camera2_serial, camera3_serial)
        
        timer.update('filter')
        
        # Find consistent matches across all three views (optional, enhances quality)
        consistent_triplets = self.find_consistent_triplets(
            kpts1, kpts2, kpts3, 
            matches_1_2, matches_1_3, matches_2_3,
            confidence_1_2, confidence_1_3, confidence_2_3
        )
        
        if len(consistent_triplets) > 0:
            print(f"Found {len(consistent_triplets)} consistent triplets across all three cameras")
            results['consistent_triplets'] = consistent_triplets
        
        timer.update('consistency')
        
        # Store descriptors for potential reuse
        desc1 = feats1['descriptors'][0].cpu().numpy()
        desc2 = feats2['descriptors'][0].cpu().numpy()
        desc3 = feats3['descriptors'][0].cpu().numpy()
        
        # Store results for each pair with full data needed for MonoDepth integration
        pair_1_2 = {
            'keypoints1': kpts1.tolist(),
            'keypoints2': kpts2.tolist(),
            'descriptors1': desc1.tolist(),
            'descriptors2': desc2.tolist(),
            'matches': matches_1_2.tolist(),
            'confidence': confidence_1_2.tolist(),
            'num_matches': int(np.sum(matches_1_2 > -1))
        }
        
        pair_1_3 = {
            'keypoints1': kpts1.tolist(),
            'keypoints3': kpts3.tolist(),
            'descriptors1': desc1.tolist(),
            'descriptors3': desc3.tolist(),
            'matches': matches_1_3.tolist(),
            'confidence': confidence_1_3.tolist(),
            'num_matches': int(np.sum(matches_1_3 > -1))
        }
        
        pair_2_3 = {
            'keypoints2': kpts2.tolist(),
            'keypoints3': kpts3.tolist(),
            'descriptors2': desc2.tolist(),
            'descriptors3': desc3.tolist(),
            'matches': matches_2_3.tolist(),
            'confidence': confidence_2_3.tolist(),
            'num_matches': int(np.sum(matches_2_3 > -1))
        }
        
        results['pairs'][f'{camera1_serial}_{camera2_serial}'] = pair_1_2
        results['pairs'][f'{camera1_serial}_{camera3_serial}'] = pair_1_3
        results['pairs'][f'{camera2_serial}_{camera3_serial}'] = pair_2_3
        
        # Generate visualizations if needed
        vis_1_2 = self.create_visualization(image1_undist, image2_undist, kpts1, kpts2, 
                                           matches_1_2, confidence_1_2, f'Matches {camera1_serial}-{camera2_serial}')
        vis_1_3 = self.create_visualization(image1_undist, image3_undist, kpts1, kpts3, 
                                           matches_1_3, confidence_1_3, f'Matches {camera1_serial}-{camera3_serial}')
        vis_2_3 = self.create_visualization(image2_undist, image3_undist, kpts2, kpts3, 
                                           matches_2_3, confidence_2_3, f'Matches {camera2_serial}-{camera3_serial}')
        
        timer.update('visualize')
        
        return results, [vis_1_2, vis_1_3, vis_2_3], timer
        
    def find_consistent_triplets(self, kpts1, kpts2, kpts3, matches_1_2, matches_1_3, matches_2_3, conf_1_2, conf_1_3, conf_2_3):
        """Find keypoints that are consistently matched across all three views"""
        triplets = []
        
        # Get valid matches
        valid_1_2 = matches_1_2 > -1
        valid_1_3 = matches_1_3 > -1
        
        # For each keypoint in the first image
        for idx1 in range(len(kpts1)):
            # Check if it matches to both camera 2 and camera 3
            if valid_1_2[idx1] and valid_1_3[idx1]:
                idx2 = matches_1_2[idx1]
                idx3 = matches_1_3[idx1]
                
                # Check if camera 2 and 3 also match with each other at these points
                if idx2 < len(matches_2_3) and matches_2_3[idx2] == idx3:
                    # This is a consistent triplet
                    confidence = min(conf_1_2[idx1], conf_1_3[idx1], conf_2_3[idx2])
                    
                    triplet = {
                        'point1': kpts1[idx1].tolist(),
                        'point2': kpts2[idx2].tolist(),
                        'point3': kpts3[idx3].tolist(),
                        'confidence': float(confidence)
                    }
                    triplets.append(triplet)
        
        # Sort by confidence
        triplets.sort(key=lambda x: x['confidence'], reverse=True)
        return triplets
    
    def create_visualization(self, img1, img2, kpts1, kpts2, matches, confidence, title):
        """Create visualization for matched features"""
        valid = matches > -1
        mkpts1 = kpts1[valid]
        mkpts2 = kpts2[matches[valid]]
        color = cm.jet(confidence[valid])
        
        text = [
            title,
            f'Keypoints: {len(kpts1)}:{len(kpts2)}',
            f'Matches: {len(mkpts1)}'
        ]
        
        return make_matching_plot_fast(
            img1, img2, kpts1, kpts2, mkpts1, mkpts2, color, text,
            path=None, show_keypoints=True)

def load_image_paths(directory, image_glob=['*.png', '*.jpg', '*.jpeg']):
    """Load image paths from a directory with glob patterns"""
    image_paths = []
    for pattern in image_glob:
        image_paths.extend(sorted(Path(directory).glob(pattern)))
    return image_paths

def load_image(path):
    """Load image from path"""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    return image

def main():
    parser = argparse.ArgumentParser(
        description='Three-Camera SuperGlue Matching',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Input/output options
    parser.add_argument('--camera1_dir', type=str, required=True,
                        help='Directory containing camera 1 images')
    parser.add_argument('--camera2_dir', type=str, required=True,
                        help='Directory containing camera 2 images')
    parser.add_argument('--camera3_dir', type=str, required=True,
                        help='Directory containing camera 3 images')
    parser.add_argument('--camera1_serial', type=str, default='24394830',
                        help='Serial number for camera 1 (reference camera)')
    parser.add_argument('--camera2_serial', type=str, default='24394835',
                        help='Serial number for camera 2')
    parser.add_argument('--camera3_serial', type=str, default='24394836',
                        help='Serial number for camera 3')
    parser.add_argument('--output_dir', type=str, default='matches_output',
                        help='Directory to save results')
    parser.add_argument('--calibration_file', type=str, default='multi_camera_calibration.json',
                        help='Path to camera calibration file')
    parser.add_argument('--skip', type=int, default=1,
                        help='Skip every N images')
    parser.add_argument('--max_length', type=int, default=1000000,
                        help='Maximum number of image triplets to process')
    parser.add_argument('--resize', type=int, nargs='+', default=[640, 480],
                        help='Resize images before processing')
    parser.add_argument('--image_glob', type=str, nargs='+', default=['*.tiff', '*.tif'],
                        help='Glob patterns for image files')
    parser.add_argument('--verify_cameras', action='store_true',
                        help='Show sample images to verify camera assignments')
    
    # SuperPoint/SuperGlue options
    parser.add_argument('--superglue', choices={'indoor', 'outdoor'}, default='outdoor',
                        help='SuperGlue weights')
    parser.add_argument('--max_keypoints', type=int, default=2000,
                        help='Maximum number of keypoints (-1 to keep all)')
    parser.add_argument('--keypoint_threshold', type=float, default=0.005,
                        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument('--nms_radius', type=int, default=4,
                        help='SuperPoint Non Maximum Suppression radius')
    parser.add_argument('--sinkhorn_iterations', type=int, default=20,
                        help='Number of Sinkhorn iterations in SuperGlue')
    parser.add_argument('--match_threshold', type=float, default=0.2,
                        help='SuperGlue match threshold')
    
    # Visualization options
    parser.add_argument('--show_visualizations', action='store_true',
                        help='Show visualizations of matches')
    parser.add_argument('--save_visualizations', action='store_true',
                        help='Save visualizations to output directory')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force CPU processing even if CUDA is available')
    
    opt = parser.parse_args()
    
    # Configure output directory
    output_dir = Path(opt.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Configure device
    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print(f'Running inference on device "{device}"')
    

    
    # Configure SuperPoint/SuperGlue
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    
    # Initialize matcher
    matcher = TrinocularMatcher(config, device)
    
    # Load calibration if available
    if opt.calibration_file:
        matcher.load_calibration(opt.calibration_file)
    
    # Load image paths
    cam1_paths = load_image_paths(opt.camera1_dir, opt.image_glob)
    cam2_paths = load_image_paths(opt.camera2_dir, opt.image_glob)
    cam3_paths = load_image_paths(opt.camera3_dir, opt.image_glob)
    
    # Print some info about the images
    print(f"Found {len(cam1_paths)} images for camera 1")
    print(f"Found {len(cam2_paths)} images for camera 2")
    print(f"Found {len(cam3_paths)} images for camera 3")
    
    # Ensure all cameras have the same number of images
    min_images = min(len(cam1_paths), len(cam2_paths), len(cam3_paths))
    if min_images == 0:
        raise ValueError("No images found in one or more camera directories")
    
    print(f"Will process {min_images} image triplets")
    
    # Limit to max_length
    num_to_process = min(min_images, opt.max_length)
    
    # Create results file
    results_file = output_dir / "matching_results.json"
    all_results = {}
    
    # Process image triplets
    for idx in range(0, num_to_process, opt.skip):
        try:
            print(f"\nProcessing triplet {idx}/{num_to_process}")
            
            # Load TIFF images and convert to grayscale
            img1 = load_tiff_rg8(cam1_paths[idx])
            img2 = load_tiff_rg8(cam2_paths[idx])
            img3 = load_tiff_rg8(cam3_paths[idx])
            
            if img1 is None or img2 is None or img3 is None:
                print(f"Failed to load one or more images for triplet {idx}")
                continue
            
            # Print original shapes for debugging
            print(f"Original image shapes: {img1.shape}, {img2.shape}, {img3.shape}")
            
            # Resize if needed
            if len(opt.resize) == 2:
                img1 = cv2.resize(img1, (opt.resize[0], opt.resize[1]))
                img2 = cv2.resize(img2, (opt.resize[0], opt.resize[1]))
                img3 = cv2.resize(img3, (opt.resize[0], opt.resize[1]))
                print(f"Resized image shapes: {img1.shape}, {img2.shape}, {img3.shape}")
            
            # Process triplet
            results, visualizations, timer = matcher.process_image_triplet(
                img1, img2, img3, 
                opt.camera1_serial, opt.camera2_serial, opt.camera3_serial
            )
            
            # Add file paths to results
            results['image_paths'] = {
                opt.camera1_serial: str(cam1_paths[idx]),
                opt.camera2_serial: str(cam2_paths[idx]),
                opt.camera3_serial: str(cam3_paths[idx])
            }
            
            # Store results
            frame_key = f"frame_{idx:06d}"
            all_results[frame_key] = results
            
            # Save intermediate results
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            # Show visualizations if requested
            if opt.show_visualizations:
                for i, (pair_name, vis) in enumerate([
                    (f"{opt.camera1_serial}-{opt.camera2_serial}", visualizations[0]),
                    (f"{opt.camera1_serial}-{opt.camera3_serial}", visualizations[1]),
                    (f"{opt.camera2_serial}-{opt.camera3_serial}", visualizations[2])
                ]):
                    cv2.namedWindow(f"Matches {pair_name}", cv2.WINDOW_NORMAL)
                    cv2.imshow(f"Matches {pair_name}", vis)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Interrupted by user")
                    break
            
            # Save visualizations if requested
            if opt.save_visualizations:
                for i, (pair_name, vis) in enumerate([
                    (f"{opt.camera1_serial}-{opt.camera2_serial}", visualizations[0]),
                    (f"{opt.camera1_serial}-{opt.camera3_serial}", visualizations[1]),
                    (f"{opt.camera2_serial}-{opt.camera3_serial}", visualizations[2])
                ]):
                    vis_path = output_dir / f"vis_{frame_key}_{pair_name}.jpg"
                    cv2.imwrite(str(vis_path), vis)
            
            # Print timing info
            timer.print()
            
        except Exception as e:
            print(f"Error processing triplet {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save final results
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nProcessed {len(all_results)} image triplets")
    print(f"Results saved to {results_file}")
    print("Done!")
    
    # Clean up
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
