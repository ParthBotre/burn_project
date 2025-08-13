#!/usr/bin/env python3
"""
CORRECTED burn classification script - fixes the inverted class mapping
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
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
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    img_resized = img.resize(target_size)
    img_tensor = torch.from_numpy(np.array(img_resized)).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor, original_size, img_resized

def classify_burn_image_fixed(image_path, output_dir="burn_results_fixed"):
    """Classify a burn image with CORRECTED class mapping"""
    print(f"🔍 Analyzing burn image: {image_path}")
    print(f"⚠️  Using CORRECTED class mapping to fix inverted classifications")
    
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
        
        # 🔧 FIX: Swap the class mapping to correct the inversion
        corrected_predictions = predictions.copy()
        corrected_predictions[predictions == 1] = 2  # Swap class 1 and 2
        corrected_predictions[predictions == 2] = 1
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create corrected colored prediction mask
        colors = {
            0: [0, 0, 0],      # Black for background
            1: [0, 255, 0],    # Green for healthy skin (FIXED)
            2: [255, 0, 0]     # Red for burn areas (FIXED)
        }
        
        corrected_colored_mask = np.zeros((*corrected_predictions.shape, 3), dtype=np.uint8)
        for class_id, color in colors.items():
            corrected_colored_mask[corrected_predictions == class_id] = color
        
        # Convert to PIL image
        mask_img = Image.fromarray(corrected_colored_mask)
        mask_img = mask_img.resize(original_size)
        
        # Create overlay (original + corrected prediction)
        original_img = Image.open(image_path).convert('RGB')
        overlay = Image.blend(original_img, mask_img, 0.3)
        
        # Save results
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save corrected prediction mask
        mask_path = os.path.join(output_dir, f"{base_name}_CORRECTED_prediction_{timestamp}.png")
        mask_img.save(mask_path)
        
        # Save corrected overlay
        overlay_path = os.path.join(output_dir, f"{base_name}_CORRECTED_overlay_{timestamp}.png")
        overlay.save(overlay_path)
        
        # Calculate statistics with CORRECTED predictions
        unique, counts = np.unique(corrected_predictions, return_counts=True)
        total_pixels = corrected_predictions.size
        
        # Create detailed report
        report_path = os.path.join(output_dir, f"{base_name}_CORRECTED_report_{timestamp}.txt")
        with open(report_path, 'w') as f:
            f.write(f"BURN CLASSIFICATION REPORT (CORRECTED)\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Image: {image_path}\n")
            f.write(f"Original size: {original_size}\n")
            f.write(f"⚠️  NOTE: Class mapping was corrected to fix inverted classifications\n\n")
            
            f.write("CORRECTED Classification Results:\n")
            f.write("-" * 40 + "\n")
            
            class_names = ['Background', 'Healthy Skin', 'Burn Areas']
            for class_id, count in zip(unique, counts):
                percentage = (count / total_pixels) * 100
                f.write(f"{class_names[class_id]}: {count:,} pixels ({percentage:.1f}%)\n")
            
            f.write(f"\nFiles saved:\n")
            f.write(f"- CORRECTED Prediction mask: {mask_path}\n")
            f.write(f"- CORRECTED Overlay image: {overlay_path}\n")
            f.write(f"- CORRECTED Report: {report_path}\n")
        
        # Print corrected summary
        print(f"\n🎯 CORRECTED Classification Results:")
        print(f"   Background: {counts[0]:,} pixels ({(counts[0]/total_pixels)*100:.1f}%)")
        print(f"   Healthy Skin: {counts[1]:,} pixels ({(counts[1]/total_pixels)*100:.1f}%)")
        print(f"   Burn Areas: {counts[2]:,} pixels ({(counts[2]/total_pixels)*100:.1f}%)")
        
        print(f"\n💾 CORRECTED Results saved to: {output_dir}/")
        print(f"   - CORRECTED Prediction mask: {os.path.basename(mask_path)}")
        print(f"   - CORRECTED Overlay image: {os.path.basename(overlay_path)}")
        print(f"   - CORRECTED Report: {os.path.basename(report_path)}")
        
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

def main():
    parser = argparse.ArgumentParser(description='CORRECTED Burn Classification Tool')
    parser.add_argument('--image', type=str, required=True, help='Path to burn image file')
    parser.add_argument('--output', type=str, default='burn_results_fixed', help='Output directory for results')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        return
    
    classify_burn_image_fixed(args.image, args.output)

if __name__ == "__main__":
    main() 