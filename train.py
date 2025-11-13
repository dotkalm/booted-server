from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')
    
    results = model.train(
        data='config.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device='mps',  # ← Use Apple's Metal Performance Shaders
        project='results',
        name='rb600_wheel_detection'
    )

if __name__ == '__main__':
    main()
