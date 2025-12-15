from multiprocessing import Process
from service.pose_service import PoseService
from labels.ucf101_labels import UCF101_LABELS,HMDB51_LABELS

# Config + checkpoint paths for PoseC3D (ucf101)
CONFIG_PATH = r"E:\PyCharmProjects\DEUTICS-GLOBAL\behaviour\test_action\mmaction2\configs\skeleton\posec3d\slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_ucf101-split1-keypoint.py"
CHECKPOINT_PATH = r"E:\PyCharmProjects\DEUTICS-GLOBAL\behaviour\test_action\action_checkpoints\slowonly_kinetics400-pretrained-ucf101.pth"

# Config + checkpoint paths for PoseC3D (hmdb51) + Replace the label file with HMDB51_LABELS
# CONFIG_PATH = r"E:\PyCharmProjects\DEUTICS-GLOBAL\behaviour\test_action\mmaction2\configs\skeleton\posec3d\slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_hmdb51-split1-keypoint.py"
# CHECKPOINT_PATH = r"E:\PyCharmProjects\DEUTICS-GLOBAL\behaviour\test_action\action_checkpoints\posec3d_hmdb51_checkpoint.pth"



def run_service_instance(model_path, source, stream_id, labels, config_path, checkpoint_path):
    service = PoseService(
        model_path=model_path,
        source=source,
        stream_id=stream_id,
        action_labels=labels,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=None  # auto-select GPU if available, else CPU
    )
    service.run()

if __name__ == "__main__":
    # Define multiple streams here
    streams = [
        {"source": "rtsp://media.camzify.live:8554/74", "model": "yolo11n-pose.pt", "stream_id": "1"},
        # {"source": "test_videos/test_gif.gif", "model": "yolo11n-pose.pt", "stream_id": "1"},
    ]

    processes = []
    for s in streams:
        p = Process(
            target=run_service_instance,
            args=(
                s["model"],          # YOLO model path
                s["source"],         # video source
                s["stream_id"],      # stream identifier
                UCF101_LABELS,       # action labels for PoseC3D
                CONFIG_PATH,         # PoseC3D config
                CHECKPOINT_PATH      # PoseC3D checkpoint
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()