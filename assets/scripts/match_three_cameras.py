import argparse
import torch
import numpy as np
import cv2
import os
import csv
from models.matching import Matching
from models.utils import (frame2tensor, make_matching_plot_fast)

# Initialize SuperGlue with default config
config = {
    'superpoint': {'nms_radius': 4, 'keypoint_threshold': 0.005, 'max_keypoints': -1},
    'superglue': {'weights': 'outdoor', 'sinkhorn_iterations': 20, 'match_threshold': 0.2}
}
matching = Matching(config).eval().cuda()  # Use GPU if available

def extract_features(image):
    """Extract SuperPoint keypoints and descriptors from an image."""
    tensor_image = frame2tensor(image, 'cuda')
    pred = matching.superpoint({'image': tensor_image})
    return {k: v[0].cpu().numpy() for k, v in pred.items()}

def match_keypoints(desc1, desc2, kpts1, kpts2):
    """Find matches between two sets of descriptors using SuperGlue."""
    data = {
        'descriptors0': torch.from_numpy(desc1).cuda().unsqueeze(0),
        'descriptors1': torch.from_numpy(desc2).cuda().unsqueeze(0),
        'keypoints0': torch.from_numpy(kpts1).cuda().unsqueeze(0),
        'keypoints1': torch.from_numpy(kpts2).cuda().unsqueeze(0),
    }
    pred = matching.superglue(data)
    matches = pred['matches0'][0].cpu().numpy()
    valid = matches > -1
    return kpts1[valid], kpts2[matches[valid]]

def match_three_images(imgs1, imgs2, imgs3):
    """Find keypoints common across three sets of images."""
    for img1, img2, img3 in zip(imgs1, imgs2, imgs3):
        features1, features2, features3 = map(extract_features, [img1, img2, img3])
        
        # Match keypoints between image pairs
        kpts1_2, kpts2 = match_keypoints(features1['descriptors'], features2['descriptors'], features1['keypoints'], features2['keypoints'])
        kpts2_3, kpts3 = match_keypoints(features2['descriptors'], features3['descriptors'], features2['keypoints'], features3['keypoints'])
        
        # Find common keypoints across all three images
        common_2 = {tuple(k): i for i, k in enumerate(kpts2)}
        common_matches = [(k1, k2, k3) for k1, k2 in zip(kpts1_2, kpts2) if tuple(k2) in common_2 for k3 in [kpts3[common_2[tuple(k2)]]]]
        
        if common_matches:
            yield zip(*common_matches)
        else:
            yield ([], [], [])

def load_images_from_directory(directory):
    """Load all images from a given directory."""
    image_paths = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('jpg', 'png', 'jpeg'))])
    return [cv2.imread(img_path) for img_path in image_paths], image_paths

def save_matches_to_csv(output_file, image_name, kpts1, kpts2, kpts3):
    """Save matched keypoints to a CSV file."""
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        for p1, p2, p3 in zip(kpts1, kpts2, kpts3):
            writer.writerow([image_name, p1.tolist(), p2.tolist(), p3.tolist()])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir1', type=str, help='Path to first camera directory')
    parser.add_argument('dir2', type=str, help='Path to second camera directory')
    parser.add_argument('dir3', type=str, help='Path to third camera directory')
    parser.add_argument('--output', type=str, default='matches.csv', help='Output CSV file for matched keypoints')
    args = parser.parse_args()
    
    imgs1, paths1 = load_images_from_directory(args.dir1)
    imgs2, paths2 = load_images_from_directory(args.dir2)
    imgs3, paths3 = load_images_from_directory(args.dir3)
    
    with open(args.output, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'Keypoint1', 'Keypoint2', 'Keypoint3'])
    
    for i, (kpts1, kpts2, kpts3) in enumerate(match_three_images(imgs1, imgs2, imgs3)):
        image_name = os.path.basename(paths1[i])
        print(f"Image set {i} ({image_name}): Found {len(kpts1)} common keypoints across three images")
        save_matches_to_csv(args.output, image_name, kpts1, kpts2, kpts3)
    
if __name__ == '__main__':
    main()
