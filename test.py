from ultralytics import YOLO
from PIL import Image
import os

MODEL_PATH = 'C:\\Users\\Ankan\\Desktop\\wall\\runs\\segment\\wall_pillar_model_v3_yolov8m\\weights\\best.pt'
TEST_IMAGE_PATH = 'C:\\Users\\Ankan\\Desktop\\wall\\my_room.jpg'

def main():
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Error: Test image not found at '{TEST_IMAGE_PATH}'")
        print("Please put an image in your project folder and update the TEST_IMAGE_PATH variable.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    print(f"Running inference on '{TEST_IMAGE_PATH}'...")
    
    results = model(TEST_IMAGE_PATH, conf=0.25)

    result = results[0]

    original_image = result.orig_img
    
    if result.masks is None:
        print("No objects (walls, floors, ceilings) were detected in the image.")
        return

    print(f"Detected {len(result.masks)} objects in total.")

    class_names = result.names
    print(f"Class names: {class_names}")

    image_with_masks = result.plot()

    output_path = 'test_result5.jpg'
    print(f"Saving result image to '{output_path}'...")
    
    Image.fromarray(image_with_masks[..., ::-1]).save(output_path)

    print("--- Done ---")
    print(f"Check your folder for '{output_path}' to see the detected walls.")

if __name__ == '__main__':
    main()