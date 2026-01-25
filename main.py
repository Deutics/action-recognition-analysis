"""Multistream Fall Detection System with Display"""

import time
from stream.multistream_manager import MultiStreamManager
from core.fall_detector import FallDetectionProcessor
from utils.logger import get_logger
from config.posture_config import PostureConfig

logger = get_logger(__name__)


def main():
    stream_sources = {
        "camera_2": "videos/yt_fail1.mp4",
        "camera_3": "videos/yt_fail2.mp4",
    }
    config=PostureConfig()
    lying_threshold = config.lying_duration_threshold
    
    enable_display = True
    
    logger.info("=== Multistream Fall Detection System ===")
    logger.info(f"Configured {len(stream_sources)} streams")
    logger.info(f"Display enabled: {enable_display}")
    
    stream_manager = MultiStreamManager(stream_sources, enable_display=enable_display)
    fall_detector = FallDetectionProcessor(
        model_path="yolo26n-pose.pt",
        lying_threshold=lying_threshold,
        confidence_threshold=0.3
    )
    
    try:
        stream_manager.start_all()
        logger.info("All streams started, beginning fall detection processing")
        
        time.sleep(1)
        
        while True:
            frames = stream_manager.get_all_latest_frames()
            
            if not frames:
                time.sleep(0.01)
                continue
            
            for stream_id, frame in frames.items():
                annotated_frame, notifications = fall_detector.process_frame(frame, stream_id)
                
                stream_manager.update_annotated_frame(stream_id, annotated_frame)
            
            if enable_display:
                if not stream_manager.display_all_streams():
                    logger.info("Display quit requested")
                    break
            
    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        stream_manager.stop_all()
        logger.info("System shutdown complete")


if __name__ == "__main__":
    main()
