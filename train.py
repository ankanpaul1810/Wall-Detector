from ultralytics import YOLO

def main():
    model = YOLO('yolov8m-seg.pt')
    data_yaml_path = 'C:/Users/Ankan/Desktop/wall/yolov8 wall detection.v1i.yolov8/data.yaml'
    run_name = 'wall_pillar_model_v3_yolov8m' 

    print(f"Starting model training on: {data_yaml_path}")
    print(f"Using model: yolov8m-seg.pt")

    results = model.train(
        data=data_yaml_path,
        epochs=300,
        imgsz=640,
        batch=4,        
        name=run_name,
        plots=True,      
        patience=50      
    )

    print("Training finished.")
    print(f"Model saved to: {results.save_dir}")

if __name__ == '__main__':
    main()