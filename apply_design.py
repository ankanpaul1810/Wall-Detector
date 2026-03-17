import cv2
import numpy as np
from ultralytics import YOLO

def clean_mask(mask_np):
    if mask_np is None or mask_np.size == 0:
        return mask_np
        
    kernel_size = 7 
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed_mask = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
    
    blurred_mask = cv2.GaussianBlur(closed_mask, (5, 5), 0) 
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(blurred_mask)
    if num_labels <= 1:
        return np.zeros_like(mask_np) 
        
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    final_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    
    return final_mask





def apply_wall_design(room_image, design_image, wall_mask):
    contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("WARN: No contours found for this mask.")
        return room_image
    
    wall_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(wall_contour, True)
    approx_corners = cv2.approxPolyDP(wall_contour, epsilon, True)

    if len(approx_corners) != 4:
        print(f"WARN: Could not find 4 corners (found {len(approx_corners)}), using bounding box.")
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
            print("WARN: findTransformECC failed or returned None, using getPerspectiveTransform.")
            warp_matrix = cv2.getPerspectiveTransform(src_corners, ordered_dest_corners)
    except cv2.error as e:
        print(f"WARN: findTransformECC failed with error: {e}. Using getPerspectiveTransform.")
        warp_matrix = cv2.getPerspectiveTransform(src_corners, ordered_dest_corners)
    
    if warp_matrix is None:
        print("ERROR: Could not compute perspective transform.")
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
    MODEL_PATH = 'C:\\Users\\Ankan\\Desktop\\wall\\runs\\segment\\wall_pillar_model_v2_small\\weights\\best.pt' 
    
    TARGET_CLASS_IDS = [2, 3]  

    DESIGN_IMAGE_PATH = 'C:\\Users\\Ankan\\Desktop\\wall\\wallpaper1.jpg'
    ROOM_IMAGE_PATH = 'C:\\Users\\Ankan\\Desktop\\wall\\my_room.jpg'
    
    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    design_image = cv2.imread(DESIGN_IMAGE_PATH)
    room_image = cv2.imread(ROOM_IMAGE_PATH)

    if design_image is None or room_image is None:
        print(f"Error: Could not read {DESIGN_IMAGE_PATH} or {ROOM_IMAGE_PATH}")
        return

    print("Running inference...")
    results = model(room_image, conf=0.5) 

    output_image = room_image.copy() 
    detections_found = 0

    for result in results:
        if result.masks is None:
            continue
            
        for i, (mask, box) in enumerate(zip(result.masks.data, result.boxes.data)):
            detected_class_id = int(box[5])
            
            if detected_class_id in TARGET_CLASS_IDS:
                detections_found += 1
                class_name = model.names[detected_class_id]
                print(f"Found {class_name} (ID: {detected_class_id}). Processing...")
                
                wall_mask_np = mask.cpu().numpy().astype(np.uint8) * 255
                wall_mask_np = cv2.resize(wall_mask_np, (output_image.shape[1], output_image.shape[0]))
                
                print("Cleaning mask edges...")
                cleaned_mask_np = clean_mask(wall_mask_np) 
                
                if np.sum(cleaned_mask_np) == 0: 
                    print("WARN: Mask was removed during cleaning process. Skipping.")
                    continue

                print("Applying design with cleaned mask...")
                output_image = apply_wall_design(output_image, design_image, cleaned_mask_np)

    if detections_found > 0:
        output_filename = 'room_with_design.jpg'
        cv2.imwrite(output_filename, output_image)
        print(f"Success! Applied design to relevant surfaces.")
        print(f"Saved result as '{output_filename}'")
    else:
        print("Could not find any walls or pillars in the image.")

if __name__ == '__main__':
    main()