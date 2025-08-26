from PIL import Image
import numpy as np
import torch
import os
from scipy.ndimage import gaussian_filter
import sys

# Add project path to sys.path
sys.path.append('Face_Frontalization')
import network

# Load the trained model
model_path = 'output/final/netG_final_2500.pt'
netG = torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"), weights_only=False)
netG.to('cuda' if torch.cuda.is_available() else 'cpu')
netG.eval()  # Set to evaluation mode

# Define output directory
output_dir = 'output/final3'
os.makedirs(output_dir, exist_ok=True)

# Function to generate and save a single image
def generate_and_test_image(profile_path):
    # Load and preprocess the profile image
    profile = Image.open(profile_path).convert('RGB').resize((128, 128), Image.Resampling.LANCZOS)
    # Save the input image before normalization
    input_filename = os.path.basename(profile_path).replace('.', '_input.')
    profile.save(f'{output_dir}/{input_filename}')

    profile = np.array(profile).transpose((2, 0, 1)) / 127.5 - 1.0  # Normalize to [-1, 1]
    profile = torch.from_numpy(profile).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float32)

    # Generate noise
    noise = torch.randn(1, 100, 1, 1).to('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate image with conditional input
    with torch.no_grad():
        generated = netG(profile, noise)

    # Convert to image, apply Gaussian blur, and save
    generated = (generated.squeeze(0).cpu().numpy().transpose((1, 2, 0)) + 1.0) * 127.5  # Denormalize
    generated = np.clip(generated, 0, 255).astype(np.uint8)
    # Apply Gaussian blur to reduce noise
    generated_blurred = np.stack([gaussian_filter(generated[:,:,i], sigma=0.5) for i in range(3)], axis=2).astype(np.uint8)
    result = Image.fromarray(generated_blurred)
    generated_filename = os.path.basename(profile_path).replace('.', '_generated_1500.')
    result.save(f'{output_dir}/{generated_filename}')
    print(f"Generated image saved as {output_dir}/{generated_filename}")

# Example usage with a test image
test_image_path = 'Dataset_Face_Frontalization/Dataset_Face_Frontalization/001/2.png'
if os.path.exists(test_image_path):
    generate_and_test_image(test_image_path)
    print(netG)
else:
    print(f"Error: Test image {test_image_path} not found. Please check the path.")