from multiprocessing import Process
from typing import Optional

from config.detection_config import ForceIntentConfig, DoorROIConfig, PostureConfig
from service.unified_detection_service_simplified import UnifiedDetectionService
from utils.logger import get_logger
from utils.roi_selector import ROISelector

logger = get_logger(__name__)


def run_service(video_source: str, source_id: str, door_roi: Optional[DoorROIConfig],
                enable_fall: bool = True, enable_force: bool = True):
    """Run detection service for a single stream."""
    try:
        force_config = None
        if enable_force:
            force_config = ForceIntentConfig(enabled=True, door_roi=door_roi)
        
        service = UnifiedDetectionService(
            video_source=video_source,
            source_id=source_id,
            posture_config=PostureConfig(),
            force_intent_config=force_config,
            enable_posture=enable_fall,
            enable_force_intent=enable_force
        )
        service.run()
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"[Main] Error in {source_id}: {str(e)}", exc_info=True)


def main():
    """Main entry point."""
    streams = [
        {
            "source": "videos/door.mp4",
            "source_id": "camera_1",
            "enable_fall": True,
            "enable_force": True
        },
    ]
    
    processes = []
    
    for stream_config in streams:
        source = stream_config["source"]
        source_id = stream_config["source_id"]
        enable_fall = stream_config.get("enable_fall", True)
        enable_force = stream_config.get("enable_force", True)
        
        door_roi = None
        if enable_force:
            door_roi = ROISelector.select_door_roi(source, source_id)
        
        p = Process(
            target=run_service,
            args=(source, source_id, door_roi, enable_fall, enable_force),
            name=f"Service-{source_id}"
        )
        p.start()
        processes.append(p)
    
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()


if __name__ == "__main__":
    main()
