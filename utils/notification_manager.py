import time
from utils.logger import get_logger

logger = get_logger(__name__)


class NotificationManager:
    def __init__(self, camera_id, allowed_actions, persist_seconds=5, fps=10):
        """
        Initialize the notification manager.

        Args:
            camera_id (int | str): Camera identifier.
            allowed_actions (list[str]): Actions that can trigger notifications.
            persist_seconds (int): Seconds an action must persist before triggering.
            fps (int): Frames per second of the video stream.
        """

        self.camera_id = camera_id
        self.allowed_actions = allowed_actions
        self.persist_seconds = persist_seconds
        self.counter = 0
        self.current_action = None

        logger.info(
            f"NotificationManager created for camera={camera_id} | "
            f"allowed_actions={len(allowed_actions)} | persist={persist_seconds}s"
        )

    def update(self, preds, annotated_frame):
        """
        Update notification state based on predictions.

        Args:
            preds (list[dict]): Model predictions with "class" and "score".
            annotated_frame (np.ndarray): Frame with annotations.

        Returns:
            np.ndarray | None: Frame to save if notification triggered, else None.
        """

        if not preds:
            if self.current_action is not None:
                logger.debug("No predictions → Resetting notification state.")
            self.counter = 0
            self.current_action = None
            return None

        top = preds[0]
        action = top["class"]

        # Ignore actions not in allowed list
        if action not in self.allowed_actions:
            if self.current_action is not None:
                logger.debug(
                    f"Ignoring action '{action}' (not allowed). Resetting persistence."
                )
            self.counter = 0
            self.current_action = None
            return None

        # Persistence tracking
        if action != self.current_action:
            logger.debug(
                f"New action detected '{action}'. Starting persistence counter."
            )
            self.current_action = action
            self.counter = 1
        else:
            self.counter += 1
            logger.debug(
                f"Action '{action}' persisting... counter={self.counter}/{self.persist_seconds}"
            )

        # Trigger notification
        if self.counter >= self.persist_seconds:
            logger.info(
                f"[NOTIFICATION TRIGGERED] camera={self.camera_id} | action='{action}'"
            )
            self.counter = 0
            return annotated_frame  # return frame to save

        return None
