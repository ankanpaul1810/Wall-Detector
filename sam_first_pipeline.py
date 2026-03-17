import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch
import os

# --- 1. CONFIGURATION ---
YOLO_MODEL_PATH = 'C:/Users/Ankan/Desktop/wall/runs/segment/wall_pillar_model_v2_small/weights/best.pt'
SAM_MODEL_PATH = 'sam_vit_l_0b3195.pth'
SAM_MODEL_TYPE = "vit_l" 
TARGET_CLASS_IDS = [2, 3]  # Targets 'pillar' and 'wall'

# --- 2. YOUR INPUT/OUTPUT FILES ---
DESIGN_IMAGE_PATH = 'C:/Users/Ankan/Desktop/wall/wallpaper1.jpg'
ROOM_IMAGE_PATH = 'C:/Users/Ankan/Desktop/wall/my_room.jpg'
FINAL_OUTPUT_IMAGE = 'room_with_design_SAM_FIRST.jpg'

# --- 3. TUNING PARAMETERS ---
# Ignore tiny masks. Adjust this area threshold as needed.
MIN_MASK_AREA = 50000 
# Confidence for YOLO's classification
YOLO_CONFIDENCE = 0.25 
# -------------------------

def apply_wall_design(room_image, design_image, wall_mask, detection_index):
    contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"WARN: No contours found for mask {detection_index}.")
        return room_image
    
    wall_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(wall_contour, True)
    approx_corners = cv2.approxPolyDP(wall_contour, epsilon, True)

    if len(approx_corners) != 4:
        print(f"WARN: Could not find 4 corners for mask {detection_index} (found {len(approx_corners)}), using bounding box.")
        x, y, w, h = cv2.boundingRect(wall_contour)
        approx_corners = np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]], dtype=np.int32)

    dest_corners = np.array([c[0] for c in approx_corners], dtype=np.float32)
    sum_corners = dest_corners.sum(axis=1)
    tl = dest_corners[np.argmin(sum_corners)]
    br = dest_corners[np.argmax(sum_corners)]
    diff_corners = np.diff(dest_corners, axis=1)
    tr = dest_corners[np.argmin(diff_corners)]
    bl = dest_corners[np.argmax(diff_corners)]
    ordered_dest_corners = np.array([tl, tr, br, bl], dtype=np.float32)

    h, w = design_image.shape[:2]
    src_corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)

    try:
        warp_matrix, _ = cv2.findTransformECC(src_corners, ordered_dest_corners, warp_mode=cv2.MOTION_HOMOGRAPHY, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001))
        if warp_matrix is None:
            warp_matrix = cv2.getPerspectiveTransform(src_corners, ordered_dest_corners)
    except cv2.error:
        warp_matrix = cv2.getPerspectiveTransform(src_corners, ordered_dest_corners)
    
    if warp_matrix is None:
        print(f"ERROR: Could not compute perspective transform for mask {detection_index}.")
        return room_image

    warped_design = cv2.warpPerspective(design_image, warp_matrix, (room_image.shape[1], room_image.shape[0]))
    final_mask_for_blending = np.zeros_like(room_image[:, :, 0], dtype=np.uint8) 
    cv2.fillPoly(final_mask_for_blending, [np.int32(ordered_dest_corners)], 255)

    mask_inv = cv2.bitwise_not(final_mask_for_blending)
    room_bg = cv2.bitwise_and(room_image, room_image, mask=mask_inv)
    design_fg = cv2.bitwise_and(warped_design, warped_design, mask=final_mask_for_blending)
    output_image = cv2.add(room_bg, design_fg)
    
    return output_image

def main():
    print("--- Starting Automated 'SAM-First' Pipeline ---")
    
    for f in [YOLO_MODEL_PATH, SAM_MODEL_PATH, DESIGN_IMAGE_PATH, ROOM_IMAGE_PATH]:
        if not os.path.exists(f):
            print(f"Error: File not found: {f}")
            return

    print("Loading models (this may take a moment)...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    yolo_model = YOLO(YOLO_MODEL_PATH)
    
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_MODEL_PATH)
    sam.to(device=device)
    # to get ALL masks
    mask_generator = SamAutomaticMaskGenerator(sam)
    print("Models loaded successfully.")

    print("Loading images...")
    design_image = cv2.imread(DESIGN_IMAGE_PATH)
    room_image_bgr = cv2.imread(ROOM_IMAGE_PATH)
    room_image_rgb = cv2.cvtColor(room_image_bgr, cv2.COLOR_BGR2RGB) 

    output_image = room_image_bgr.copy()
    detections_found = 0

    # --- Step 1: SAM Generates All Masks ---
    print("Step 1: SAM is generating all masks... (This will be slow)")
    # 'generate' returns a list of dictionaries, each 'mask' is one object
    all_masks = mask_generator.generate(room_image_rgb)
    print(f"SAM found {len(all_masks)} total masks.")

    # Sort masks by area (largest first)
    all_masks = sorted(all_masks, key=lambda x: x['area'], reverse=True)

    # --- Step 2: YOLO Classifies Each Mask ---
    print(f"Step 2: Classifying masks with YOLO (min area: {MIN_MASK_AREA})...")
    
    for i, mask_data in enumerate(all_masks):
        mask_area = mask_data['area']
        
        # Filter 1: Ignore masks that are too small
        if mask_area < MIN_MASK_AREA:
            # print(f"  Mask {i} skipped (too small: {mask_area})")
            continue
            
        # Filter 2: Ignore masks that are the *entire* image
        if mask_area > (room_image_bgr.shape[0] * room_image_bgr.shape[1] * 0.95):
            print(f"  Mask {i} skipped (too large: {mask_area})")
            continue
            
        print(f"\nAnalyzing Mask {i} (Area: {mask_area})...")
        
        # Get the bounding box of the SAM mask
        x, y, w, h = mask_data['bbox']
        # Create a cropped image from the bounding box
        cropped_img = room_image_bgr[y:y+h, x:x+w]
        
        # Run YOLO on this *specific crop*
        yolo_results = yolo_model(cropped_img, conf=YOLO_CONFIDENCE, verbose=False)
        
        found_target = False
        for result in yolo_results:
            if result.boxes is None:
                continue
            
            for box in result.boxes:
                detected_class_id = int(box.cls[0])
                if detected_class_id in TARGET_CLASS_IDS:
                    class_name = yolo_model.names[detected_class_id]
                    print(f"  > SUCCESS: YOLO identified this mask as '{class_name}' (conf: {box.conf[0]:.2f})")
                    found_target = True
                    break
            if found_target:
                break
        
        # --- Step 3: Apply Design ---
        if found_target:
            detections_found += 1
            
            # Get the full-resolution boolean mask from SAM
            sam_mask_bool = mask_data['segmentation']
            # Convert it to a 0-255 image
            sam_mask_np = (sam_mask_bool * 255).astype(np.uint8)
            
            # Save a debug image
            cv2.imwrite(f'debug_sam_first_mask_{detections_found}.jpg', sam_mask_np)
            
            print(f"Step 3: Applying design to clean SAM mask {detections_found}...")
            output_image = apply_wall_design(output_image, design_image, sam_mask_np, detections_found)

    # --- Final Output ---
    if detections_found > 0:
        cv2.imwrite(FINAL_OUTPUT_IMAGE, output_image)
        print(f"\n--- Pipeline Complete ---")
        print(f"Successfully applied design to {detections_found} surfaces.")
        print(f"Final image saved as: {FINAL_OUTPUT_IMAGE}")
    else:
        print("\n--- Pipeline Complete ---")
        print(f"Could not find any masks that YOLO classified as 'wall' or 'pillar'.")

if __name__ == '__main__':
    main()