import cv2


def draw_label(frame, text, pos=(20,40), color=(0,255,0)):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

def draw_status(frame, buffer_len, window, pos=(20,340)):
    status = f"Buffer: {buffer_len}/{window}"
    cv2.putText(frame, status, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

def draw_keypoints(frame, kp_xy, kp_score, threshold=0.5):
    for person_idx in range(kp_xy.shape[0]):
        for joint_idx, (x, y) in enumerate(kp_xy[person_idx]):
            if kp_score[person_idx, joint_idx] > threshold:
                cv2.circle(frame, (int(x), int(y)), 3, (0,255,0), -1)
    return frame

def init_video_writer(output_path, frame_shape, fps=25):
    h, w = frame_shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (w, h))
