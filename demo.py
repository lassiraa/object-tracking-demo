import cv2
import torch
import torchvision.transforms as transforms
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.utils import draw_bounding_boxes


def evaluate(mod, inp):
    return mod(inp)


cap = cv2.VideoCapture(0)
# Step 1: Initialize model with the best available weights
weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights, box_score_thresh=0.9).to(torch.float32)
model.eval()
tracker = DeepSort(max_age=5)

preprocess = weights.transforms()
transform = transforms.ToTensor()
transform_bounding_boxes = lambda x: (x[0], x[1], x[2]-x[0], x[3]-x[1])

while(True):
    ret, frame = cap.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = torch.tensor(frame, dtype=torch.uint8).permute(2, 0, 1)

    preprocessed_frame = preprocess(frame)

    prediction = evaluate(model, [preprocessed_frame])[0]
    bbs = [
        (transform_bounding_boxes(boxes.tolist()), scores.item(), weights.meta["categories"][labels.item()])
        for boxes, scores, labels in
        zip(prediction["boxes"], prediction["scores"], prediction["labels"])
    ]
    tracks = tracker.update_tracks(bbs, frame=frame.permute(1, 2, 0).numpy())
    track_boxes = torch.zeros((0,4))
    track_labels = []
    for i, track in enumerate(tracks):
        if not track.is_confirmed():
            continue
        track_boxes = torch.cat([track_boxes, torch.tensor([track.to_ltrb()])])
        track_labels.append(f"{track.det_class} {track.track_id}")
    #labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    print(track_boxes)
    print(track_labels)
    box = draw_bounding_boxes(
        frame,
        boxes=track_boxes,
        labels=track_labels,
        colors="red",
        width=4,
        font_size=30
    )
    im = box.detach().permute(1, 2, 0).numpy()
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

    cv2.imshow('frame', im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()