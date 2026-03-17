import cv2
import numpy as np
import torch
import os
import argparse  
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch.nn.functional as F
from collections import Counter

SAM_MODEL_PATH = "sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"
SEGFORMER_MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

ALLOW_CLASSES = [
    "wall", "column", "panel", "building"
]
ALLOW_CLASS_IDS = [] 
BLOCKLIST_CLASSES = [
    "ceiling", "floor", "sky", "person", "chair", "table", "sofa",
    "potted_plant", "plant", "bed", "monitor", "screen", "computer",
    "book", "car", "bicycle", "lamp", "light"
]
BLOCKLIST_CLASS_IDS = []
MIN_MASK_AREA = 50000
MAX_IMAGE_WIDTH = 1280

def make_texture_seamless(texture, detail_boost=True):
    h, w = texture.shape[:2]
    texture = texture.astype(np.float32)
    blend_width = w // 6
    blend_height = h // 6
    left = texture[:, :blend_width]
    right = texture[:, -blend_width:]
    mix_lr = cv2.addWeighted(left, 0.8, cv2.flip(right, 1), 0.2, 0)
    texture[:, :blend_width] = mix_lr
    texture[:, -blend_width:] = cv2.flip(mix_lr, 1)
    top = texture[:blend_height, :]
    bottom = texture[-blend_height:, :]
    mix_tb = cv2.addWeighted(top, 0.8, cv2.flip(bottom, 0), 0.2, 0)
    texture[:blend_height, :] = mix_tb
    texture[-blend_height:, :] = cv2.flip(mix_tb, 0)
    if detail_boost:
        blur = cv2.GaussianBlur(texture, (3, 3), 0)
        texture = cv2.addWeighted(texture, 1.2, blur, -0.2, 0)
    return np.clip(texture, 0, 255).astype(np.uint8)

def tile_texture(texture, width, height):
    texture = make_texture_seamless(texture.copy(), detail_boost=True)
    th, tw = texture.shape[:2]
    rep_x = int(np.ceil(width / tw))
    rep_y = int(np.ceil(height / th))
    tiled = np.tile(texture, (rep_y, rep_x, 1))
    return tiled[:height, :width]

def adjust_lighting_uniform(design, room, mask, strength=0.15):
    room_gray = cv2.cvtColor(room, cv2.COLOR_BGR2GRAY)
    design_gray = cv2.cvtColor(design, cv2.COLOR_BGR2GRAY)
    mask_pixels = room_gray[mask > 0]
    if mask_pixels.size == 0:
        return design 
    room_mean = np.mean(mask_pixels)
    design_mean = np.mean(design_gray)
    if design_mean > 0:
        scale = room_mean / design_mean
        design = np.clip(design * scale, 0, 255).astype(np.uint8)
    mean_color_pixels = design[mask > 0]
    if mean_color_pixels.size > 0:
        mean_color = np.mean(mean_color_pixels, axis=0)
        design = cv2.addWeighted(design, 1 - strength, np.full_like(design, mean_color, dtype=np.uint8), strength, 0)
    return design

def feather_mask(mask, amount=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (amount, amount))
    eroded = cv2.erode(mask, kernel, iterations=2)
    blurred = cv2.GaussianBlur(eroded, (7, 7), 0)
    return blurred

def enhance_texture_clarity(texture, clarity=0.3):
    blurred = cv2.GaussianBlur(texture, (3, 3), 0)
    enhanced = cv2.addWeighted(texture, 1 + clarity, blurred, -clarity, 0)
    return np.clip(enhanced, 0, 255).astype(np.uint8)

def apply_wall_design(room, design, mask, lighting_strength, matte_strength):
    h, w = room.shape[:2]
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = (mask > 0).astype(np.uint8) * 255
    mask_feathered = feather_mask(mask, 5)
    design_tiled = tile_texture(design, w, h)
    design_tiled = adjust_lighting_uniform(design_tiled, room, mask, strength=lighting_strength)
    design_tiled = enhance_texture_clarity(design_tiled, clarity=matte_strength)
    mask_f = mask_feathered.astype(np.float32) / 255.0
    mask_f = cv2.merge([mask_f, mask_f, mask_f])
    blended = design_tiled * mask_f + room * (1 - mask_f)
    return np.clip(blended, 0, 255).astype(np.uint8)

def run_pipeline(args):
    """
    Main pipeline logic, now takes the 'args' object.
    """
    global ALLOW_CLASS_IDS, BLOCKLIST_CLASS_IDS
    print("Starting SAM + SegFormer pipeline")
    for f in [SAM_MODEL_PATH, args.design, args.room]:
        if not os.path.exists(f):
            print(f"Error: Missing file: {f}")
            return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("Loading SegFormer model...")
    processor = AutoImageProcessor.from_pretrained(SEGFORMER_MODEL_NAME)
    segformer_model = AutoModelForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL_NAME).to(device)
    ALLOW_CLASS_IDS = []
    BLOCKLIST_CLASS_IDS = []
    allow_classes_set = set(ALLOW_CLASSES)
    blocklist_classes_set = set(BLOCKLIST_CLASSES)
    for class_name, class_id in segformer_model.config.label2id.items():
        if class_name in allow_classes_set:
            ALLOW_CLASS_IDS.append(class_id)
        if class_name in blocklist_classes_set:
            BLOCKLIST_CLASS_IDS.append(class_id)
            
    print(f"Allowing {len(ALLOW_CLASS_IDS)} class IDs for: {ALLOW_CLASSES}")
    print(f"Blocking {len(BLOCKLIST_CLASS_IDS)} class IDs for: {BLOCKLIST_CLASSES}")

    print("Loading SAM model")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_MODEL_PATH)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=12,
        pred_iou_thresh=0.90,     
        stability_score_thresh=0.90
    )
    print("Models loaded.")
    design_img = cv2.imread(args.design)
    room_bgr = cv2.imread(args.room)
    if room_bgr is None:
        print(f"Error: Could not read room image from {args.room}")
        return
    if design_img is None:
        print(f"Error: Could not read design image from {args.design}")
        return
    room_pil = Image.open(args.room).convert("RGB")
    room_np = np.array(room_pil)
    room_total_pixels = room_np.shape[0] * room_np.shape[1]

    if room_np.shape[1] > MAX_IMAGE_WIDTH:
        new_w = MAX_IMAGE_WIDTH
        new_h = int(room_np.shape[0] * (MAX_IMAGE_WIDTH / room_np.shape[1])) 
        print(f"Resizing image from {room_np.shape[1]}x{room_np.shape[0]} to {new_w}x{new_h}")
        room_np = cv2.resize(room_np, (new_w, new_h))
        room_bgr = cv2.resize(room_bgr, (new_w, new_h))

    print("Running SegFormer...")
    inputs = processor(images=room_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = segformer_model(**inputs)

    logits = F.interpolate(outputs.logits, size=room_np.shape[:2], mode="bilinear", align_corners=False)
    predicted_labels_map = logits.argmax(1).squeeze().cpu().numpy()

    print("Running SAM...")
    all_masks = mask_generator.generate(room_np)
    all_masks = sorted(all_masks, key=lambda x: x["area"], reverse=True)

    output = room_bgr.copy()
    detected = 0

    print(f"Found {len(all_masks)} potential masks. Filtering...")
    
    allow_id_set = set(ALLOW_CLASS_IDS)
    blocklist_id_set = set(BLOCKLIST_CLASS_IDS)    
    for i, mask_data in enumerate(all_masks):
        area = mask_data["area"]
        if area < MIN_MASK_AREA:
            continue
        mask_bool = mask_data["segmentation"]
        labels_in_mask = predicted_labels_map[mask_bool]
        if labels_in_mask.size == 0:
            continue
        label_counts = Counter(labels_in_mask).most_common(5)        
        decision = "uncertain" 
        top_classes = []
        
        for label_id, count in label_counts:
            top_classes.append(segformer_model.config.id2label.get(label_id, "unknown"))
            if label_id in blocklist_id_set:
                decision = "blocked"
                break   
            if label_id in allow_id_set:
                decision = "allowed"
                break 
        if decision == "allowed":
            detected += 1
            print(f"Applying design to mask {i} (Top Classes: {top_classes}, Area: {area}) <-- ALLOWED")
            mask_img = (mask_bool * 255).astype(np.uint8)
            output = apply_wall_design(
                output, 
                design_img, 
                mask_img, 
                args.lighting, 
                args.matte
            )
            if args.single_wall:
                print("Single wall mode enabled. Stopping after first wall.")
                break
        else:
             print(f"Skipping mask {i} (Top Classes: {top_classes}, Area: {area}) <-- {decision.upper()}")
    if detected > 0:
        cv2.imwrite(args.output, output)
        print(f"Wallpaper applied successfully to {detected} regions.")
        print(f"Saved as {args.output}")
    else:
        print("No suitable (allowed) wall regions found.")
        cv2.imwrite(args.output, room_bgr)
        print("Saving original image as output since no walls were detected.")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Wallpaper Visualizer Pipeline")
    parser.add_argument("--room", type=str, required=True, help="Path to the room image")
    parser.add_argument("--design", type=str, required=True, help="Path to the design/wallpaper image")
    parser.add_argument("--output", type=str, required=True, help="Path to save the final output image")
    parser.add_argument("--lighting", type=float, default=0.2, help="Lighting adjustment strength (0.0 to 1.0)")
    parser.add_argument("--matte", type=float, default=0.3, help="Matte/Clarity effect (0.0 to 1.0)")
    parser.add_argument(
        "--single_wall", 
        action="store_true", 
        help="If set, applies design to only the largest detected wall"
    )
    args = parser.parse_args()
    run_pipeline(args)