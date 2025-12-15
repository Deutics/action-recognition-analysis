# service/pose_service.py
import os
import threading
import cv2
import time

from stream.stream_handler import StreamHandler
from detection.yolo_pose import YOLOPoseDetector
from formatter.formatter import PoseFormatter
from action_recognition.posec3d_infer import PoseC3DRecognizer
from utils.visualization import draw_keypoints, draw_label, draw_status, init_video_writer
from utils.logger import get_logger

logger = get_logger(__name__)


class PoseService:

    def __init__(
        self,
        model_path,
        source,
        stream_id,
        action_labels,
        config_path,
        checkpoint_path,
        device="cpu",
        window_size=24,
        max_persons=2,
        num_keypoints=17,
        no_person_reset = 10
    ):
        self.no_person_counter = 0
        self.NO_PERSON_RESET = no_person_reset
        self.stream_id = stream_id
        self.running = True

        self.video_writer = None
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        self.output_path = os.path.join(output_dir, f"output_{time.time_ns()}.mp4")

        self.stream = StreamHandler(source)
        self.yolo = YOLOPoseDetector(model_path, max_persons, num_keypoints)
        self.formatter = PoseFormatter(window_size, max_persons, num_keypoints)
        self.recognizer = PoseC3DRecognizer(config_path, checkpoint_path, device, action_labels)

        self.latest_frame = None
        self.latest_action = None

    def frame_loop(self):
        self.stream.start_stream()

        while self.running:
            frame, motion = self.stream.read_frame()
            if frame is None or not motion:
                continue

            self.latest_frame = frame

        self.stream.release_stream()


    def action_loop(self):
        while self.running:
            if not self.formatter.is_ready():
                time.sleep(0.01)
                continue

            sample = self.formatter.build_sample()
            result = self.recognizer.infer(sample)
            self.latest_action = self.recognizer.get_top_result(result)
            self.formatter.reset()

    def run(self):
        logger.info(f"PoseService {self.stream_id} started")

        threading.Thread(target=self.frame_loop, daemon=True).start()
        threading.Thread(target=self.action_loop, daemon=True).start()

        while self.running:
            if self.latest_frame is None:
                cv2.waitKey(1)
                continue

            frame = self.latest_frame.copy()

            # INIT VIDEO WRITER ON FIRST FRAME
            if self.video_writer is None:
                self.video_writer = init_video_writer(
                    self.output_path,
                    frame.shape,
                    fps=25
                )

            # YOLO Pose
            keypoints_xy, keypoints_conf = self.yolo.infer(frame)

            if keypoints_xy.sum() > 0:
                self.no_person_counter = 0
                self.formatter.update_buffer(keypoints_xy, keypoints_conf)
                draw_keypoints(frame, keypoints_xy, keypoints_conf)
            else:
                self.no_person_counter += 1
                if self.no_person_counter >= self.NO_PERSON_RESET:
                    self.formatter.reset()
                    self.no_person_counter = 0

            if self.latest_action:
                label, score = self.latest_action
                draw_label(frame, f"{label}: {score:.2f}")

            draw_status(
                frame,
                len(self.formatter.keypoints_xy_buffer),
                self.formatter.window_size
            )

            # WRITE FRAME (ONLY AFTER INIT)
            self.video_writer.write(frame)

            cv2.imshow(f"YOLO Pose {self.stream_id}", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                self.running = False

        if self.video_writer:
            self.video_writer.release()

        cv2.destroyAllWindows()
        logger.info(f"PoseService {self.stream_id} stopped")
