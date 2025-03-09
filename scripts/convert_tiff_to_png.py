from pathlib import Path
import cv2
import numpy as np
import argparse 
def convert_tiff_to_png(input_dir, output_dir, demosaic_method='SIMPLE', grayscale=False):
    """
    Convert TIFF RG8 files to PNG format
    
    Args:
        input_dir (str): Directory containing TIFF RG8 files
        output_dir (str): Directory to save PNG files
        demosaic_method (str): Demosaicing method to use
        grayscale (bool): Whether to convert to grayscale (for SuperGlue) or keep color
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Wipe output directory
    for file in Path(output_dir).glob('*'):
        if file.is_file():
            file.unlink()
    
    # Get all TIFF files
    tiff_files = list(Path(input_dir).glob('*.tiff')) + list(Path(input_dir).glob('*.tif'))
    
    if not tiff_files:
        print(f"No TIFF files found in {input_dir}")
        return
    
    print(f"Found {len(tiff_files)} TIFF files in {input_dir}")
    
    # Set demosaicing method
    if demosaic_method == 'SIMPLE':
        demosaic_code = cv2.COLOR_BayerRG2RGB  # For RG8 pattern
    elif demosaic_method == 'VNG':
        demosaic_code = cv2.COLOR_BayerRG2RGB_VNG
    elif demosaic_method == 'EA':
        demosaic_code = cv2.COLOR_BayerRG2RGB_EA
    else:
        print(f"Unknown demosaicing method: {demosaic_method}")
        demosaic_code = cv2.COLOR_BayerRG2RGB
    
    # Process each file
    for i, tiff_path in enumerate(tiff_files):
        try:
            # Load TIFF file
            tiff_img = cv2.imread(str(tiff_path), cv2.IMREAD_UNCHANGED)
            
            if tiff_img is None:
                print(f"Failed to load {tiff_path}")
                continue
            
            # Process image based on its format
            if len(tiff_img.shape) == 2:
                # It's a raw Bayer pattern, demosaic it to RGB
                rgb_img = cv2.cvtColor(tiff_img, demosaic_code)
                
                # Either save as RGB or convert to grayscale based on parameter
                if grayscale:
                    output_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
                else:
                    # For saving color images, convert from RGB to BGR (OpenCV's default)
                    output_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            elif len(tiff_img.shape) == 3:
                # It's already RGB/BGR format
                if grayscale:
                    output_img = cv2.cvtColor(tiff_img, cv2.COLOR_BGR2GRAY)
                else:
                    output_img = tiff_img  # Keep as is
            else:
                print(f"Unexpected image format: {tiff_img.shape}")
                continue
            
            # Normalize to 8-bit if needed
            if output_img.dtype == np.uint16:
                output_img = (output_img / 256).astype(np.uint8)
            
            # Create output filename
            output_path = Path(output_dir) / f"{tiff_path.stem}.png"
            
            # Save as PNG
            cv2.imwrite(str(output_path), output_img)
            
            # Print progress
            if (i + 1) % 10 == 0 or i == 0 or i == len(tiff_files) - 1:
                print(f"Processed {i + 1}/{len(tiff_files)} files")
            
        except Exception as e:
            print(f"Error processing {tiff_path}: {e}")
            continue
    
    print(f"Converted {len(tiff_files)} TIFF files to PNG in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Convert TIFF RG8 files to PNG format')
    
    parser.add_argument('--tiff_dir_1', type=str, required=True,
                      help='Directory containing camera 1 TIFF files')
    parser.add_argument('--tiff_dir_2', type=str, required=True,
                      help='Directory containing camera 2 TIFF files')
    parser.add_argument('--tiff_dir_3', type=str, required=True,
                      help='Directory containing camera 3 TIFF files')
    
    parser.add_argument('--png_dir_1', type=str, required=True,
                      help='Directory to save camera 1 PNG files')
    parser.add_argument('--png_dir_2', type=str, required=True,
                      help='Directory to save camera 2 PNG files')
    parser.add_argument('--png_dir_3', type=str, required=True,
                      help='Directory to save camera 3 PNG files')
    
    parser.add_argument('--demosaic_method', type=str, default='SIMPLE',
                      choices=['SIMPLE', 'VNG', 'EA'],
                      help='Demosaicing method to use')
    
    parser.add_argument('--grayscale', action='store_true',
                      help='Convert to grayscale (for SuperGlue compatibility)')
    
    args = parser.parse_args()
    
    # Convert each camera's images
    print("Converting camera 1 images...")
    convert_tiff_to_png(args.tiff_dir_1, args.png_dir_1, args.demosaic_method, args.grayscale)
    
    print("\nConverting camera 2 images...")
    convert_tiff_to_png(args.tiff_dir_2, args.png_dir_2, args.demosaic_method, args.grayscale)
    
    print("\nConverting camera 3 images...")
    convert_tiff_to_png(args.tiff_dir_3, args.png_dir_3, args.demosaic_method, args.grayscale)
    
    print("\nAll conversions complete!")