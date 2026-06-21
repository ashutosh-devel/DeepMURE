import os
import glob
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from models.SpeckleFormer import SpeckleFormer_Medium

def load_image(path):
    """Loads an image, converts to grayscale and normalizes to [0, 1]."""
    img = Image.open(path).convert('L')
    img = transforms.ToTensor()(img)
    return img

def save_image(tensor, path):
    """Saves a tensor image to a file, converting back to [0, 255] uint8."""
    img = tensor.squeeze().cpu().detach().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)

def main():
    parser = argparse.ArgumentParser(description='Evaluate SpeckleFormer on a folder of corrupted images.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint (.pth)')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing corrupted .png/.tif images')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save denoised images')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Set up device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Initialize model
    model = SpeckleFormer_Medium().to(device)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully.")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Find images
    image_paths = glob.glob(os.path.join(args.input_dir, '*.png')) + glob.glob(os.path.join(args.input_dir, '*.tif'))
    if not image_paths:
        print(f"No .png or .tif images found in {args.input_dir}")
        return

    print(f"Found {len(image_paths)} images. Processing...")

    with torch.no_grad():
        for path in image_paths:
            filename = os.path.basename(path)
            try:
                # Load and preprocess
                img_tensor = load_image(path).unsqueeze(0).to(device)
                #img_tensor = img_tensor/255.0  # Normalize to [0, 1]
                img_tensor = img_tensor**2 #Gamma noise corruption and not rayleigh
                img_tensor = torch.clamp(img_tensor, 1e-7)
                # Inference
                denoised_tensor = model(img_tensor)
                #denoised_tensor *= 255.0
                #denoised_tensor = torch.clamp(denoised_tensor, 0, 255)
                denoised_tensor = torch.sqrt(denoised_tensor) #Gamma noise corruption and not rayleigh
                # Save result
                save_path = os.path.join(args.output_dir, f"denoised_{filename}")
                save_image(denoised_tensor, save_path)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"Denoising complete. Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()
