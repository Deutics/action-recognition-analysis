# from expected_actions import VALID_KINETICS_ACTIONS

class NotificationManager:
    def __init__(self, camera_id, allowed_actions, persist_seconds=2, fps=10):
        self.camera_id = camera_id
        self.allowed_actions = allowed_actions
        self.persist_seconds = persist_seconds
        self.counter = 0
        self.current_action = None

    def update(self, preds, annotated_frame):
        if not preds:
            self.counter = 0
            self.current_action = None
            return None

        top = preds[0]
        action = top["class"]

        if action not in self.allowed_actions:
            self.counter = 0
            self.current_action = None
            return None

        # Track persistence
        if action != self.current_action:
            self.current_action = action
            self.counter = 1
        else:
            self.counter += 1

        # If persisted for X seconds -> trigger
        if self.counter >= self.persist_seconds:
            self.counter = 0
            print("____________Notification Generated_______________")
            return annotated_frame  # return frame to save

        return None
