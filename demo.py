from typing import Tuple, Union

import cv2
import torch
import torchvision.transforms as transforms
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
)
from torchvision.utils import draw_bounding_boxes


# Transforms bounding box from xmin ymin, xmax, ymax to xmin, ymin, width, height
transform_bounding_boxes = lambda x: (x[0], x[1], x[2] - x[0], x[3] - x[1])
# Gets center point of a bounding box
get_center = lambda x: (x[0] + (x[2] - x[0]) / 2, x[1] + (x[3] - x[1]) / 2)
# Calculates manhattan distance from p1 to p2
calculate_distance = lambda p1, p2: abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


class Game:
    def __init__(self, court_width: int):
        self.score = [0, 0]
        self.player = 0
        self.opponent = 1
        self.winner = None
        self.court_width = court_width
        self.recent_locations = []
        self.min_occurences = 10

    def decide_point(self, landing_zone: int):
        if landing_zone == self.opponent:
            self.score[self.player] += 1
        else:
            self.score[self.opponent] += 1
        self.player, self.opponent = self.opponent, self.player
        self.recent_locations = []

    def check_winner(self) -> Union[int, None]:
        if (max(self.score) > 20 and abs(self.score[0] - self.score[1]) > 1) or max(
            self.score
        ) == 30:
            self.winner = int(self.score[1] > self.score[0])
            return self.winner
        return None

    def calculate_traversed_distance(self):
        traversed_distance = 0
        prev_location = self.recent_locations[0]
        for location in self.recent_locations:
            traversed_distance += calculate_distance(location, prev_location)
            prev_location = location
        return traversed_distance

    def update_game(self, location: Tuple[int]) -> Union[int, None]:
        if self.winner is not None:
            return self.winner
        center = get_center(location)
        if len(self.recent_locations) == self.min_occurences:
            del self.recent_locations[0]
        self.recent_locations.append(center)
        landing_zone = int(center[0] >= (self.court_width // 2))
        traversed_distance = self.calculate_traversed_distance()
        if (
            len(self.recent_locations) == self.min_occurences
            and traversed_distance <= 15
        ):
            self.decide_point(landing_zone)
            if self.check_winner() is not None:
                return self.winner
            return None


# Object class to track for game
GAME_OBJECT_CLASS = "person"

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

    ret, frame = cap.read()
    resolution = frame.shape

    game = Game(resolution[1])

    while True:
        ret, frame = cap.read()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_tensor = torch.tensor(
            frame_rgb, dtype=torch.uint8, device=device
        ).permute(2, 0, 1)

        preprocessed_frame = preprocess(frame_tensor)

        prediction = model([preprocessed_frame])[0]

        game_object_bbs = []
        for boxes, score, label in zip(
            prediction["boxes"], prediction["scores"], prediction["labels"]
        ):
            class_name = weights.meta["categories"][label.item()]
            if class_name != GAME_OBJECT_CLASS:
                continue
            game_object_bbs.append(
                (
                    transform_bounding_boxes(boxes.tolist()),
                    score.item(),
                    class_name,
                )
            )

        tracks = tracker.update_tracks(game_object_bbs, frame=frame)

        track_boxes = torch.zeros((0, 4), device=device)
        track_labels = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            obj_location = track.to_ltrb()
            track_id = track.track_id
            track_boxes = torch.cat(
                [track_boxes, torch.tensor(obj_location.reshape((1, 4)), device=device)]
            )
            game.update_game(obj_location)
            track_labels.append(f"Track: {track_id}")

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

        # Draw a line to the image for visual clarity
        im[:, resolution[1] // 2, :] = 1
        game_status = (
            f"Game score: {game.score}"
            if game.winner is None
            else f"Winner is player {game.winner}"
        )
        im = cv2.putText(
            im, game_status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0)
        )

        cv2.imshow("frame", im)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
