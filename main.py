"""
Main Entry Point for Posture Detection Service
Handles multiprocess stream processing
"""

from multiprocessing import Process

from config.posture_config import PostureConfig
from service.posture_recognition_service import PoseRecognition
from utils.logger import get_logger
logger = get_logger(__name__)


def run_posture_service(source: str, source_id: str):
    """
    Run posture detection service for a single stream
    
    Args:
        source: Video source (file path or RTSP URL)
        source_id: Unique identifier for stream
    """
    try:
        service = PoseRecognition(
            video_source=source,
            source_id=source_id,
            model_path="yolo11m-pose.pt",
            config=PostureConfig(),
            confidence_threshold=0.5
        )
        service.run()
    except Exception as e:
        logger.error(f"Error running service for source {source_id}: {str(e)}")




def main():
    """Main entry point - setup and run services"""
    
    # Define video streams to process
    streams = [
        # {"source": "rtsp://media.camzify.live:8554/73", "source_id": "stream_1"},
        # {"source": "rtsp://media.camzify.live:8554/74", "source_id": "stream_2"},
        # {"source": 0, "source_id": "stream_2"},
        # Uncomment for local video file testing
        {"source": "videos/yt_fail1.mp4", "source_id": "local_video"},
        # {"source": "videos/googleimg12.jpg", "source_id": "local_video"},
    ]

    logger.info(f"Starting {len(streams)} posture detection service(s)")
    
    # Start a process for each stream
    processes = []
    for stream_config in streams:
        logger.info(f"Starting process for {stream_config['source_id']}")
        
        p = Process(
            target=run_posture_service,
            args=(stream_config["source"], stream_config["source_id"]),
            name=f"PoseService-{stream_config['source_id']}"
        )
        p.start()
        processes.append(p)
    
    logger.info(f"All {len(processes)} processes started")
    
    # Wait for all processes to complete
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        logger.info("Interrupt received, terminating all processes...")
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
        logger.info("All processes terminated")


if __name__ == "__main__":
    main()
