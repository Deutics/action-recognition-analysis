import cv2
from datetime import datetime

def overlay_predictions(frame, preds, fps, max_display=3):
    h, w = frame.shape[:2]
    y = 30

    # ---------- Timestamp (top-right corner) ----------
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Calculate text width for right alignment
    (text_width, text_height), _ = cv2.getTextSize(
        f"Time: {timestamp}",
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        2
    )

    # Position → 10px margin from right
    x_pos = w - text_width - 10

    cv2.putText(frame,
                f"Time: {timestamp}",
                (x_pos, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2)

    # ---------- Predictions (top-left as before) ----------
    y_pred = 30
    for p in preds[:max_display]:
        cv2.putText(frame,
                    f"{p['class']} ({p['score']:.3f})",
                    (10, y_pred),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
        y_pred += 25

    # ---------- FPS below predictions ----------
    cv2.putText(frame,
                f"FPS: {fps:.2f}",
                (10, y_pred),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2)

    return frame
