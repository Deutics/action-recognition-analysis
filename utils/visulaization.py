import cv2
from datetime import datetime
from utils.logger import get_logger

logger = get_logger(__name__)


def overlay_predictions(frame, predictions, fps, max_display=3):
    """
    Overlay predictions, timestamp, and FPS on a video frame.

    Args:
        frame (np.ndarray): Input frame (BGR format).
        predictions (list[dict]): Predictions with "class" and "score".
        fps (float): Frames per second value.
        max_display (int): Maximum number of predictions to display.

    Returns:
        np.ndarray: Annotated frame with overlays.
    """

    logger.debug("Rendering overlay for current frame...")

    if not predictions:
        logger.debug("No predictions received for overlay.")
        predictions = []

    if fps <= 0:
        logger.warning(f"Overlay received suspicious FPS value: {fps}")

    height, width = frame.shape[:2]

    # ------------------------------------------------------
    # Timestamp (top-right corner)
    # ------------------------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp_text = f"Time: {timestamp}"

    (timestamp_width, timestamp_height), _ = cv2.getTextSize(
        timestamp_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.7,
        thickness=2
    )

    timestamp_x_pos = width - timestamp_width - 10
    timestamp_y_pos = 30

    cv2.putText(
        frame,
        timestamp_text,
        (timestamp_x_pos, timestamp_y_pos),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # ------------------------------------------------------
    # Predictions (top-left)
    # ------------------------------------------------------
    prediction_y_pos = 30
    displayed_predictions = predictions[:max_display]

    if displayed_predictions:
        logger.debug(f"Overlaying {len(displayed_predictions)} predictions.")
    else:
        logger.debug("No predictions available to overlay.")

    for pred in displayed_predictions:
        text = f"{pred['class']} ({pred['score']:.3f})"
        cv2.putText(
            frame,
            text,
            (10, prediction_y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
        prediction_y_pos += 25

    # ------------------------------------------------------
    # FPS (below predictions)
    # ------------------------------------------------------
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(
        frame,
        fps_text,
        (10, prediction_y_pos),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    return frame
