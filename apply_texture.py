"""
AI Texture Application Script
Applies a texture from one image onto the walls of a room from another image
while preserving 3D geometry and perspective using SDXL + ControlNet + IP-Adapter
"""

import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL
)
from diffusers.utils import load_image
from transformers import DPTForDepthEstimation, DPTImageProcessor


def get_depth_map(image, device):
    """
    Generate a depth map from an input image using DPT model.
    
    Args:
        image: PIL Image of the room
        device: torch device (cuda/cpu)
    
    Returns:
        PIL Image: Normalized depth map as 3-channel RGB image
    """
    print("Loading depth estimation model...")
    depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(device)
    feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
    
    print("Processing image for depth estimation...")
    # Prepare image for depth estimation
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    
    # Predict depth
    with torch.no_grad():
        outputs = depth_estimator(**inputs)
        predicted_depth = outputs.predicted_depth
    
    # Interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    
    # Normalize the depth map
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)
    
    # Convert to 3-channel RGB image for ControlNet
    depth_map = Image.new("RGB", depth.size)
    depth_map.paste(depth)
    
    print("Depth map generated successfully!")
    return depth_map


def main():
    """Main execution function"""
    print("="*60)
    print("AI Texture Application System")
    print("="*60)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    if device == "cpu":
        print("WARNING: CUDA not available. This will be very slow!")
    
    # Define input files
    room_image_path = "room.jpg"  # Room image (content)
    texture_image_path = "wallpaper1.jpg"  # Texture image (style)
    
    print(f"\nLoading images...")
    print(f"  Room image: {room_image_path}")
    print(f"  Texture image: {texture_image_path}")
    
    try:
        room_image = load_image(room_image_path)
        texture_image = load_image(texture_image_path)
        print("Images loaded successfully!")
    except Exception as e:
        print(f"ERROR: Failed to load images: {e}")
        return
    
    # Generate depth map from room image
    print("\n" + "="*60)
    print("Step 1: Generating Depth Map")
    print("="*60)
    control_image = get_depth_map(room_image, device)
    
    # Save depth map for visualization
    control_image.save("depth_map.jpg")
    print("Depth map saved as 'depth_map.jpg'")
    
    # Load ControlNet model
    print("\n" + "="*60)
    print("Step 2: Loading ControlNet Model")
    print("="*60)
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    print("ControlNet loaded!")
    
    # Load VAE
    print("\n" + "="*60)
    print("Step 3: Loading VAE")
    print("="*60)
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=torch.float16
    )
    print("VAE loaded!")
    
    # Load main pipeline
    print("\n" + "="*60)
    print("Step 4: Loading SDXL Pipeline (This may take a while...)")
    print("="*60)
    pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16"
    ).to(device)
    
    # Enable memory optimizations
    pipeline.enable_model_cpu_offload()
    print("Pipeline loaded successfully!")
    
    # Load IP-Adapter for style transfer
    print("\n" + "="*60)
    print("Step 5: Loading IP-Adapter")
    print("="*60)
    pipeline.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter-plus_sdxl_vit-h.safetensors"
    )
    print("IP-Adapter loaded!")
    
    # Set generation parameters
    print("\n" + "="*60)
    print("Step 6: Configuring Generation Parameters")
    print("="*60)
    
    prompt = "a photorealistic image of an office waiting room, with a new floral wallpaper, high quality, detailed, professional"
    negative_prompt = "blurry, low quality, ugly, cartoon, cgi, watermark, text, deformed, extra limbs, bad anatomy, distorted, unrealistic"
    
    controlnet_conditioning_scale = 0.9  # High value to preserve room geometry
    ip_adapter_scale = 0.7  # High value for prominent texture style
    num_inference_steps = 30
    seed = 42
    
    print(f"  Prompt: {prompt}")
    print(f"  ControlNet Scale: {controlnet_conditioning_scale}")
    print(f"  IP-Adapter Scale: {ip_adapter_scale}")
    print(f"  Inference Steps: {num_inference_steps}")
    print(f"  Seed: {seed}")
    
    # Set IP-Adapter scale
    pipeline.set_ip_adapter_scale(ip_adapter_scale)
    
    # Generate the image
    print("\n" + "="*60)
    print("Step 7: Generating Image (This will take several minutes...)")
    print("="*60)
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    try:
        output = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_image,  # Depth map for ControlNet
            ip_adapter_image=texture_image,  # Texture image for IP-Adapter
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            height=1024,
            width=1024
        )
        
        result_image = output.images[0]
        
        # Save the result
        output_path = "final_room_with_texture.jpg"
        result_image.save(output_path)
        
        print("\n" + "="*60)
        print("SUCCESS!")
        print("="*60)
        print(f"✓ Output image saved as '{output_path}'")
        print(f"✓ Depth map saved as 'depth_map.jpg'")
        print("\nTexture application complete!")
        
    except Exception as e:
        print(f"\nERROR during generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
