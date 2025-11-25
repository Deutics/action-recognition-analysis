# main.py
from multiprocessing import Process
from service.action_recognition_service import ActionRecognitionService


def run_action_service(model_name, source, class_csv):
    """
    IMPORTANT:
    This runs INSIDE the child process.
    The model will be loaded inside ActionRecognitionService(),
    so nothing is pickled. No lambdas, no PyTorch modules, nothing.
    """
    service = ActionRecognitionService(
        model_name=model_name,
        video_source=source,
        class_csv=class_csv,
    )
    service.run()


if __name__ == "__main__":
    # Required for Windows multiprocessing
    streams = [
        # {"source": "https://media.camzify.live:8888/73/index.m3u8", "model": "c2d_r50"},
        {"source": "https://media.camzify.live:8888/74/index.m3u8", "model": "c2d_r50"},
        # {"source": 0, "model": "c2d_r50"},
    ]

    processes = []

    for s in streams:
        p = Process(
            target=run_action_service,
            args=(s["model"], s["source"], "all_kinetics_class_mapping/kinetics_400_labels.csv")
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
