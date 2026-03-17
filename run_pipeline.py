import cv2
import numpy as np
import torch
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch.nn.functional as F
from collections import Counter

# ---------------- CONFIGURATION ----------------
SAM_MODEL_PATH = "sam_vit_b_01ec64.pth"   # ✅ smaller model
SAM_MODEL_TYPE = "vit_b"                  # ✅ update type
SEGFORMER_MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
TARGET_CLASSES = ["wall", "column"]
DESIGN_IMAGE_PATH = "C:\\Users\\Ankan\\Desktop\\wall\\wallpapers\\wallpaper2.jpg"
ROOM_IMAGE_PATH = "C:\\Users\\Ankan\\Desktop\\wall\\my_room6.jpg"
FINAL_OUTPUT_IMAGE = "C:\\Users\\Ankan\\Desktop\\wall\\room_with_design_FINAL6.jpg"
MIN_MASK_AREA = 50000
MAX_IMAGE_WIDTH = 1280  # resize large inputs to save VRAM
# ------------------------------------------------


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
    room_mean = np.mean(room_gray[mask > 0])
    design_mean = np.mean(design_gray)
    if design_mean > 0:
        scale = room_mean / design_mean
        design = np.clip(design * scale, 0, 255).astype(np.uint8)
    mean_color = np.mean(design, axis=(0, 1))
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


def apply_wall_design(room, design, mask):
    h, w = room.shape[:2]
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = (mask > 0).astype(np.uint8) * 255

    mask = feather_mask(mask, 5)
    design_tiled = tile_texture(design, w, h)
    design_tiled = adjust_lighting_uniform(design_tiled, room, mask)
    design_tiled = enhance_texture_clarity(design_tiled, clarity=0.4)

    mask_f = mask.astype(np.float32) / 255.0
    mask_f = cv2.merge([mask_f, mask_f, mask_f])
    blended = design_tiled * mask_f + room * (1 - mask_f)
    return np.clip(blended, 0, 255).astype(np.uint8)


def main():
    print("Starting SAM + SegFormer pipeline")

    for f in [SAM_MODEL_PATH, DESIGN_IMAGE_PATH, ROOM_IMAGE_PATH]:
        if not os.path.exists(f):
            print(f"Error: Missing file: {f}")
            return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    processor = AutoImageProcessor.from_pretrained(SEGFORMER_MODEL_NAME)
    segformer_model = AutoModelForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL_NAME).to(device)

    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_MODEL_PATH)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(
        sam, points_per_side=8, pred_iou_thresh=0.9, stability_score_thresh=0.9
    )

    design_img = cv2.imread(DESIGN_IMAGE_PATH)
    room_bgr = cv2.imread(ROOM_IMAGE_PATH)
    room_pil = Image.open(ROOM_IMAGE_PATH).convert("RGB")
    room_np = np.array(room_pil)

    # auto-resize large images
    if room_np.shape[1] > MAX_IMAGE_WIDTH:
        new_w = MAX_IMAGE_WIDTH
        new_h = int(room_np.shape[0] * MAX_IMAGE_WIDTH / room_np.shape[1])
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

    for i, mask_data in enumerate(all_masks):
        area = mask_data["area"]
        if area < MIN_MASK_AREA:
            continue

        mask_bool = mask_data["segmentation"]
        labels = predicted_labels_map[mask_bool]
        if labels.size == 0:
            continue
        main_label = Counter(labels).most_common(1)[0][0]
        class_name = segformer_model.config.id2label[main_label]

        if class_name in TARGET_CLASSES:
            detected += 1
            print(f"Applying design to mask {i} ({class_name}, area={area})")
            mask_img = (mask_bool * 255).astype(np.uint8)
            output = apply_wall_design(output, design_img, mask_img)

    if detected > 0:
        cv2.imwrite(FINAL_OUTPUT_IMAGE, output)
        print(f"Wallpaper applied successfully to {detected} regions.")
        print(f"Saved as {FINAL_OUTPUT_IMAGE}")
    else:
        print("No wall regions found.")


if __name__ == "__main__":
    main()
