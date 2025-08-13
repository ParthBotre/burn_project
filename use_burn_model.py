#!/usr/bin/env python3
"""
Real-world burn classification script
Use your trained model to classify new burn images
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import argparse
from datetime import datetime

class SimpleBurnModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # Fixed architecture to match the trained model
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),      # features.0
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),     # features.2
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, 3, padding=1),     # features.5
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),     # features.7
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),    # features.10
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),   # features.12
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Conv2d(128, 64, 1),               # classifier.0
            nn.ReLU(),
            nn.Conv2d(64, num_classes, 1)        # classifier.2
        )
        
        self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = self.upsample(x)
        return x

def load_model():
    """Load the trained burn classification model"""
    model_path = 'exp/burn/simple/burn_model.pth'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Please train the model first!")
    
    model = SimpleBurnModel(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def preprocess_image(image_path, target_size=(256, 256)):
    """Preprocess image for the model"""
    # Load and convert to RGB
    img = Image.open(image_path).convert('RGB')
    
    # Store original size for later
    original_size = img.size
    
    # Resize for model
    img_resized = img.resize(target_size)
    
    # Convert to tensor and normalize
    img_tensor = torch.from_numpy(np.array(img_resized)).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor, original_size, img_resized

def classify_burn_image(image_path, output_dir="burn_results"):
    """Classify a burn image and save results"""
    print(f"🔍 Analyzing burn image: {image_path}")
    
    try:
        # Load model
        model = load_model()
        print("✅ Model loaded successfully")
        
        # Preprocess image
        img_tensor, original_size, img_resized = preprocess_image(image_path)
        print(f"📏 Image preprocessed: {original_size} → {img_resized.size}")
        
        # Run inference
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            predictions = torch.argmax(output, dim=1).squeeze().numpy()
        
        print("✅ Inference completed")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create colored prediction mask
        colors = {
            0: [0, 0, 0],      # Black for background
            1: [0, 255, 0],    # Green for healthy skin
            2: [255, 0, 0]     # Red for burn areas
        }
        
        colored_mask = np.zeros((*predictions.shape, 3), dtype=np.uint8)
        for class_id, color in colors.items():
            colored_mask[predictions == class_id] = color
        
        # Convert to PIL image
        mask_img = Image.fromarray(colored_mask)
        
        # Resize back to original size
        mask_img = mask_img.resize(original_size)
        
        # Create overlay (original + prediction)
        original_img = Image.open(image_path).convert('RGB')
        overlay = Image.blend(original_img, mask_img, 0.3)
        
        # Save results
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save prediction mask
        mask_path = os.path.join(output_dir, f"{base_name}_prediction_{timestamp}.png")
        mask_img.save(mask_path)
        
        # Save overlay
        overlay_path = os.path.join(output_dir, f"{base_name}_overlay_{timestamp}.png")
        overlay.save(overlay_path)
        
        # Calculate statistics
        unique, counts = np.unique(predictions, return_counts=True)
        total_pixels = predictions.size
        
        # Create detailed report
        report_path = os.path.join(output_dir, f"{base_name}_report_{timestamp}.txt")
        with open(report_path, 'w') as f:
            f.write(f"Burn Classification Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Image: {image_path}\n")
            f.write(f"Original size: {original_size}\n\n")
            
            f.write("Classification Results:\n")
            f.write("-" * 30 + "\n")
            
            class_names = ['Background', 'Healthy Skin', 'Burn Areas']
            for class_id, count in zip(unique, counts):
                percentage = (count / total_pixels) * 100
                f.write(f"{class_names[class_id]}: {count:,} pixels ({percentage:.1f}%)\n")
            
            f.write(f"\nFiles saved:\n")
            f.write(f"- Prediction mask: {mask_path}\n")
            f.write(f"- Overlay image: {overlay_path}\n")
            f.write(f"- This report: {report_path}\n")
        
        # Print summary
        print(f"\n🎯 Classification Results:")
        print(f"   Background: {counts[0]:,} pixels ({(counts[0]/total_pixels)*100:.1f}%)")
        print(f"   Healthy Skin: {counts[1]:,} pixels ({(counts[1]/total_pixels)*100:.1f}%)")
        print(f"   Burn Areas: {counts[2]:,} pixels ({(counts[2]/total_pixels)*100:.1f}%)")
        
        print(f"\n💾 Results saved to: {output_dir}/")
        print(f"   - Prediction mask: {os.path.basename(mask_path)}")
        print(f"   - Overlay image: {os.path.basename(overlay_path)}")
        print(f"   - Detailed report: {os.path.basename(report_path)}")
        
        return {
            'background_pixels': counts[0],
            'healthy_pixels': counts[1],
            'burn_pixels': counts[2],
            'total_pixels': total_pixels,
            'burn_percentage': (counts[2]/total_pixels)*100
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def batch_classify(input_dir, output_dir="burn_results"):
    """Classify multiple images in a directory"""
    print(f"🔄 Batch processing images from: {input_dir}")
    
    # Find all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(input_dir) if f.lower().endswith(ext)])
    
    if not image_files:
        print(f"❌ No image files found in {input_dir}")
        return
    
    print(f"📁 Found {len(image_files)} images to process")
    
    results = []
    
    for i, image_file in enumerate(image_files):
        print(f"\n--- Processing {i+1}/{len(image_files)}: {image_file} ---")
        image_path = os.path.join(input_dir, image_file)
        
        result = classify_burn_image(image_path, output_dir)
        if result:
            results.append({
                'image': image_file,
                'burn_percentage': result['burn_percentage']
            })
    
    # Create batch summary
    if results:
        summary_path = os.path.join(output_dir, f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Batch Classification Summary\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total images: {len(results)}\n\n")
            
            f.write("Results by image:\n")
            f.write("-" * 40 + "\n")
            
            for result in results:
                f.write(f"{result['image']}: {result['burn_percentage']:.1f}% burn area\n")
            
            f.write(f"\nSummary saved to: {summary_path}")
        
        print(f"\n🎉 Batch processing complete!")
        print(f"📊 Summary saved to: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='Burn Classification Tool')
    parser.add_argument('--image', type=str, help='Path to single image file')
    parser.add_argument('--batch', type=str, help='Path to directory with multiple images')
    parser.add_argument('--output', type=str, default='burn_results', help='Output directory for results')
    
    args = parser.parse_args()
    
    if not args.image and not args.batch:
        print("❌ Please specify either --image or --batch")
        print("\nUsage examples:")
        print("  Single image: python use_burn_model.py --image path/to/burn_image.jpg")
        print("  Batch process: python use_burn_model.py --batch path/to/image_folder")
        print("  Custom output: python use_burn_model.py --image image.jpg --output my_results")
        return
    
    if args.image:
        # Single image classification
        if not os.path.exists(args.image):
            print(f"❌ Image not found: {args.image}")
            return
        
        classify_burn_image(args.image, args.output)
    
    elif args.batch:
        # Batch classification
        if not os.path.exists(args.batch):
            print(f"❌ Directory not found: {args.batch}")
            return
        
        batch_classify(args.batch, args.output)

if __name__ == "__main__":
    main() 