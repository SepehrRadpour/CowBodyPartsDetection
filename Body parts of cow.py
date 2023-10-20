import cv2
import torch
import torchvision.transforms as T
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn


def detect_objects(model, video_path, output_path):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model = model.to(device)
    model.eval()

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    transform = T.Compose([T.ToTensor()])

    with torch.no_grad():
        while True:
            ret, frame = video.read()
            if not ret:
                break

            frame_tensor = transform(frame).unsqueeze(0).to(device)
            predictions = model(frame_tensor)[0]

            for i in range(len(predictions['labels'])):
                label = predictions['labels'][i].cpu().item()
                score = predictions['scores'][i].cpu().item()
                box = predictions['boxes'][i].cpu().tolist()

                if score > 0.5:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255//(label/10), 0), 2)
                    cv2.putText(frame, f"Label: {label}, Score: {score:.2f}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            out.write(frame)
            cv2.imshow("Object Detection", frame)

    video.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Detection saved to {output_path}")

model = fasterrcnn_resnet50_fpn(pretrained=True)
detect_objects(model, 'input_video.mp4', 'output_video.mp4')
