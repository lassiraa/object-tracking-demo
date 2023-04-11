import cv2
import torch
import torchvision.transforms as transforms
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
)
from torchvision.utils import draw_bounding_boxes


class Game:
    def __init__(self):
        self.score = (0, 0)
        self.player = 0
        self.opponent = 1
        self.winner = None

    def decide_point(self, landing_zone: int):
        if landing_zone == self.opponent:
            self.score[self.player] += 1
        else:
            self.score[self.opponent] += 1
        self.player, self.opponent = self.opponent, self.player

    def check_winner(self):
        if (max(self.score) > 20 and abs(self.score[0] - self.score[1]) > 1) or max(
            self.score
        ) == 30:
            self.winner = int(self.score[1] > self.score[0])


transform_bounding_boxes = lambda x: (x[0], x[1], x[2] - x[0], x[3] - x[1])
get_center = lambda x: (x[0] + (x[2] - x[0]) / 2, x[1] + (x[3] - x[1]) / 2)
calculate_distance = lambda p1, p2: abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=weights, box_score_thresh=0.9
    ).to(device=device, dtype=torch.float32)
    model.eval()

    tracker = DeepSort(max_age=5)

    preprocess = weights.transforms()
    transform = transforms.ToTensor()

    cap = cv2.VideoCapture(0)
    track_locations = dict()

    while True:
        ret, frame = cap.read()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_tensor = torch.tensor(
            frame_rgb, dtype=torch.uint8, device=device
        ).permute(2, 0, 1)

        preprocessed_frame = preprocess(frame_tensor)

        prediction = model([preprocessed_frame])[0]
        bbs = [
            (
                transform_bounding_boxes(boxes.tolist()),
                scores.item(),
                weights.meta["categories"][labels.item()],
            )
            for boxes, scores, labels in zip(
                prediction["boxes"], prediction["scores"], prediction["labels"]
            )
        ]
        tracks = tracker.update_tracks(bbs, frame=frame)
        track_boxes = torch.zeros((0, 4), device=device)
        track_labels = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            cur_location = track.to_ltrb()
            track_id = track.track_id
            track_boxes = torch.cat(
                [track_boxes, torch.tensor(cur_location.reshape((1, 4)), device=device)]
            )
            prev_location = track_locations.get(track_id, [0, 0, 0, 0])
            prev_center = get_center(prev_location)
            cur_center = get_center(cur_location)
            distance = calculate_distance(prev_center, cur_center)
            track_labels.append(f"{track.det_class} {track_id} - {distance}")
            track_locations[track_id] = cur_location

        if track_labels:
            box = draw_bounding_boxes(
                frame_tensor,
                boxes=track_boxes,
                labels=track_labels,
                colors="red",
                width=4,
            )
            im = box.permute(1, 2, 0).numpy(force=True)
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        else:
            im = frame

        cv2.imshow("frame", im)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
