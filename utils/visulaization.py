# utils/visulaization.py
import cv2

def overlay_predictions(frame, preds, fps, max_display=3):
    y = 30
    for p in preds[:max_display]:
        cv2.putText(frame,
                    f"{p['class']} ({p['score']})",
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        y += 25
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return frame
