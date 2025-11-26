# notifications/action_notification.py
import cv2
import os
from datetime import datetime

class ActionNotification:
    def __init__(self, camera_id, frame, save_root="notifications"):
        self.camera_id = camera_id
        self.frame = frame
        self.save_root = save_root

    def register(self):
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

        camera_folder = os.path.join(self.save_root, f"camera_{self.camera_id}")
        os.makedirs(camera_folder, exist_ok=True)

        filepath = os.path.join(camera_folder, f"{timestamp}.jpg")
        cv2.imwrite(filepath, self.frame)

        print(f"[NOTIFICATION] Saved {filepath}")
