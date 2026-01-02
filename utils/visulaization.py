import cv2
from utils.logger import get_logger

logger = get_logger(__name__)


def overlay_predictions(frame, predictions, source_fps, model_fps, max_display=3):
    """
    Overlay predictions, source FPS, and model throughput FPS on a video frame.

    Args:
        frame (np.ndarray): Input frame (BGR format).
        predictions (list[dict]): Predictions with "class" and "score".
        source_fps (float): FPS reported by the video source (cv2 CAP_PROP_FPS).
        model_fps (float): Model throughput FPS (1 / infer_time).
        max_display (int): Maximum number of predictions to display.

    Returns:
        np.ndarray: Annotated frame with overlays.
    """

    logger.debug("Rendering overlay for current frame...")

    if not predictions:
        logger.debug("No predictions received for overlay.")
        predictions = []

    if source_fps <= 0:
        logger.warning(f"Overlay received suspicious source FPS value: {source_fps}")
    if model_fps <= 0:
        logger.warning(f"Overlay received suspicious model FPS value: {model_fps}")

    # Predictions
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


    # FPS values
    fps_text = f"Source FPS: {source_fps:.2f}"
    cv2.putText(
        frame,
        fps_text,
        (10, prediction_y_pos),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )
    prediction_y_pos += 25

    throughput_text = f"Model FPS: {model_fps:.2f}"
    cv2.putText(
        frame,
        throughput_text,
        (10, prediction_y_pos),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 255),
        2
    )

    return frame