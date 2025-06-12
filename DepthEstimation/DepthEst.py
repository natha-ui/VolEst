#!/usr/bin/env python3
"""
Food Depth Estimation Script
Estimates depth maps from single food images using monocular depth estimation.
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path

class FoodDepthEstimator:
    def __init__(self, model_type='DPT_Large'):
        """
        Initialize the depth estimator.
        
        Args:
            model_type (str): Type of MiDaS model to use
                             Options: 'DPT_Large', 'DPT_Hybrid', 'MiDaS_small'
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load pre-trained MiDaS model
        self.model = torch.hub.load('intel-isl/MiDaS', model_type)
        self.model.to(self.device)
        self.model.eval()
        
        # Load transforms
        midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
        
        if model_type == 'DPT_Large' or model_type == 'DPT_Hybrid':
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform
    
    def preprocess_image(self, image_path):
        """
        Load and preprocess image for depth estimation.
        
        Args:
            image_path (str): Path to input image
            
        Returns:
            tuple: (original_image, preprocessed_tensor)
        """
        # Load image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        input_tensor = self.transform(img).to(self.device)
        
        return img, input_tensor
    
    def estimate_depth(self, input_tensor):
        """
        Estimate depth from preprocessed image tensor.
        
        Args:
            input_tensor: Preprocessed image tensor
            
        Returns:
            numpy.ndarray: Depth map
        """
        with torch.no_grad():
            prediction = self.model(input_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=input_tensor.shape[-2:],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        # Convert to numpy
        depth_map = prediction.cpu().numpy()
        
        return depth_map
    
    def postprocess_depth(self, depth_map, original_shape):
        """
        Postprocess depth map to match original image dimensions.
        
        Args:
            depth_map: Raw depth map from model
            original_shape: Shape of original image (H, W, C)
            
        Returns:
            numpy.ndarray: Processed depth map
        """
        # Resize depth map to match original image
        depth_resized = cv2.resize(depth_map, (original_shape[1], original_shape[0]))
        
        # Normalize depth values
        depth_normalized = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return depth_normalized
    
    def create_depth_visualization(self, original_image, depth_map, colormap=cv2.COLORMAP_PLASMA):
        """
        Create visualization combining original image and depth map.
        
        Args:
            original_image: Original RGB image
            depth_map: Depth map
            colormap: OpenCV colormap for depth visualization
            
        Returns:
            numpy.ndarray: Combined visualization
        """
        # Apply colormap to depth
        depth_colored = cv2.applyColorMap(depth_map, colormap)
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
        
        # Create side-by-side visualization
        h, w = original_image.shape[:2]
        combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
        combined[:, :w] = original_image
        combined[:, w:] = depth_colored
        
        return combined, depth_colored
    
    def analyze_food_depth(self, depth_map, threshold_percentile=70):
        """
        Analyze depth characteristics specific to food images.
        
        Args:
            depth_map: Normalized depth map
            threshold_percentile: Percentile threshold for foreground detection
            
        Returns:
            dict: Analysis results
        """
        # Calculate depth statistics
        depth_stats = {
            'mean_depth': np.mean(depth_map),
            'std_depth': np.std(depth_map),
            'min_depth': np.min(depth_map),
            'max_depth': np.max(depth_map),
            'depth_range': np.max(depth_map) - np.min(depth_map)
        }
        
        # Identify foreground (likely food items) vs background
        threshold = np.percentile(depth_map, threshold_percentile)
        foreground_mask = depth_map > threshold
        
        # Calculate foreground statistics
        if np.any(foreground_mask):
            foreground_depth = depth_map[foreground_mask]
            depth_stats.update({
                'foreground_mean': np.mean(foreground_depth),
                'foreground_std': np.std(foreground_depth),
                'foreground_percentage': np.sum(foreground_mask) / depth_map.size * 100
            })
        
        return depth_stats
    
    def process_image(self, image_path, output_dir=None, save_visualization=True):
        """
        Complete pipeline to process a single food image.
        
        Args:
            image_path (str): Path to input image
            output_dir (str): Directory to save outputs
            save_visualization (bool): Whether to save visualization
            
        Returns:
            tuple: (depth_map, analysis_results, visualization)
        """
        print(f"Processing: {image_path}")
        
        # Preprocess
        original_img, input_tensor = self.preprocess_image(image_path)
        
        # Estimate depth
        raw_depth = self.estimate_depth(input_tensor)
        
        # Postprocess
        depth_map = self.postprocess_depth(raw_depth, original_img.shape)
        
        # Create visualization
        if save_visualization:
            combined_vis, depth_colored = self.create_depth_visualization(original_img, depth_map)
        else:
            combined_vis, depth_colored = None, None
        
        # Analyze depth for food-specific insights
        analysis = self.analyze_food_depth(depth_map)
        
        # Save outputs if directory specified
        if output_dir and save_visualization:
            os.makedirs(output_dir, exist_ok=True)
            base_name = Path(image_path).stem
            
            # Save depth map
            depth_path = os.path.join(output_dir, f"{base_name}_depth.png")
            cv2.imwrite(depth_path, depth_map)
            
            # Save colored depth visualization
            depth_colored_bgr = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)
            depth_colored_path = os.path.join(output_dir, f"{base_name}_depth_colored.png")
            cv2.imwrite(depth_colored_path, depth_colored_bgr)
            
            # Save combined visualization
            combined_bgr = cv2.cvtColor(combined_vis, cv2.COLOR_RGB2BGR)
            combined_path = os.path.join(output_dir, f"{base_name}_combined.png")
            cv2.imwrite(combined_path, combined_bgr)
            
            print(f"Saved outputs to: {output_dir}")
        
        return depth_map, analysis, combined_vis

def main():
    parser = argparse.ArgumentParser(description="Estimate depth for food images")
    parser.add_argument("input", help="Input image path or directory")
    parser.add_argument("--output", "-o", help="Output directory", default="./depth_outputs")
    parser.add_argument("--model", "-m", choices=['DPT_Large', 'DPT_Hybrid', 'MiDaS_small'], 
                       default='DPT_Large', help="Model type to use")
    parser.add_argument("--no-vis", action="store_true", help="Skip visualization generation")
    parser.add_argument("--show", action="store_true", help="Display results using matplotlib")
    
    args = parser.parse_args()
    
    # Initialize estimator
    estimator = FoodDepthEstimator(model_type=args.model)
    
    # Process single image or directory
    if os.path.isfile(args.input):
        image_paths = [args.input]
    elif os.path.isdir(args.input):
        # Find all image files in directory
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        for ext in extensions:
            image_paths.extend(Path(args.input).glob(f"*{ext}"))
            image_paths.extend(Path(args.input).glob(f"*{ext.upper()}"))
        image_paths = [str(p) for p in image_paths]
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        return
    
    if not image_paths:
        print("No valid image files found")
        return
    
    print(f"Found {len(image_paths)} image(s) to process")
    
    # Process each image
    for img_path in image_paths:
        try:
            depth_map, analysis, visualization = estimator.process_image(
                img_path, 
                output_dir=args.output, 
                save_visualization=not args.no_vis
            )
            
            # Print analysis
            print(f"\nDepth Analysis for {Path(img_path).name}:")
            print(f"  Depth range: {analysis['depth_range']:.1f}")
            print(f"  Mean depth: {analysis['mean_depth']:.1f}")
            print(f"  Depth std: {analysis['std_depth']:.1f}")
            if 'foreground_percentage' in analysis:
                print(f"  Foreground area: {analysis['foreground_percentage']:.1f}%")
                print(f"  Foreground mean depth: {analysis['foreground_mean']:.1f}")
            
            # Show visualization if requested
            if args.show and visualization is not None:
                plt.figure(figsize=(12, 6))
                plt.imshow(visualization)
                plt.axis('off')
                plt.title(f"Depth Estimation: {Path(img_path).name}")
                plt.tight_layout()
                plt.show()
                
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()
