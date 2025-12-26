# Posture Detection Service (YOLO Pose + Geometry-Based Classification)

This repository provides a **robust and modular Posture Detection Service** that can process video streams, live camera feeds, and image files to classify human postures in real-time.

It uses **YOLO pose estimation** for keypoint detection combined with **geometric angle analysis** and **multi-cue classification** to accurately detect postures like Standing, Sitting, Bending, Squatting, and Lying.

---

##  Key Features

-  **Multi-angle support**: Works with front-camera view, side-view, and angled camera positions
-  **Real-time processing**: Optimized for video streams and live camera feeds
-  **Automatic tilt correction**: Detects and corrects for tilted camera angles
-  **Robust classification**: Uses multiple geometric cues (knee angles, hip height, torso orientation)
-  **Modular architecture**: Easy to extend and customize
-  **Motion-based optimization**: Only processes frames with detected motion to save compute

---

##  How the System Works (Full Flow)

### **main.py**
- Reads configuration (video sources, model paths)
- Instantiates **PoseRecognition** service
- Starts processing stream with real-time display

### **service/posture_recognition_service.py**
- Orchestrates the entire pipeline:
  - Initializes YOLO pose model
  - Starts video stream handler with motion detection
  - Runs the inference + classification loop
  - Outputs posture predictions with confidence
  - Annotates frames for visualization

### **core/classifier.py**
- **Heart of the system**: Multi-stage posture classification
- **Step 1**: Extracts geometric features (angles, ratios, positions)
- **Step 2**: Detects and corrects image tilt
- **Step 3**: Priority-based classification:
  1. **Lying** (horizontal orientation)
  2. **Sitting** (bent knees + low hip - catches front-facing)
  3. **Standing** (vertical torso + straight knees)
  4. **Squatting** (moderate knee bend + medium hip height)
  5. **Bending** (vertical legs + forward torso)
- **Key innovation**: Uses **knee angles** as camera-angle-invariant features

### **preprocessing/keypoint_extractor.py**
- Extracts relevant keypoints from YOLO output
- Computes midpoints (shoulder_mid, hip_mid, knee_mid, ankle_mid)
- Validates keypoint confidence scores
- Filters low-confidence detections

### **core/geometry.py**
- **Geometric utility functions**:
  - `calculate_angle_from_horizontal()` - Segment orientation
  - `calculate_deviation_from_vertical()` - Uprightness measure
  - `calculate_angle_between_three_points()` - Joint angles (hip-knee-ankle)
  - `are_segments_aligned()` - Checks if torso and legs are parallel
- All angle calculations handle bidirectional segments

### **config/posture_config.py**
- **Centralized configuration** for all thresholds:
```python
  vertical_deviation_threshold: float = 12.0       # Max deviation to be "vertical"
  sitting_knee_angle_max: float = 120.0            # Bent knee threshold
  standing_knee_angle_min: float = 150.0           # Straight knee threshold
  sitting_hip_height_ratio_max: float = 0.45       # Hip position for sitting
  horizontal_deviation_threshold: float = 25.0     # Lying detection
```
- Easy to tune without touching code

### **service/frame_annotator.py**
- Draws posture labels on frames
- Shows confidence scores
- Displays bounding boxes with color-coded posture indicators:
  -  **Standing**: Green
  -  **Sitting**: Blue
  -  **Bending**: Orange
  -  **Squatting**: Cyan
  -  **Lying**: Magenta

### **stream/stream_handler.py**
- **Connects to video sources**:
  - RTSP streams
  - Video files (MP4, AVI, etc.)
  - Live camera (webcam)
- **Features**:
  - Motion detection (background subtraction)
  - Automatic reconnection on stream failure
  - FPS control and frame buffering
  - Only runs inference when motion detected (optimization)

### **utils/logger.py**
- Structured logging with timestamps
- Configurable log levels (DEBUG, INFO, WARNING, ERROR)
- Helps debug detection issues

---

##  Classification Logic (How Postures Are Detected)

### **The Challenge:**
Traditional angle-based methods fail when:
- Person faces camera (depth information lost)
- Camera is tilted
- Person shifts weight or moves slightly

### **Solution: Multi-Cue Priority System**
```
┌─────────────────────────────────────────────────────────┐
│  STEP 1: Extract Geometric Features                    │
│  ─────────────────────────────────────────────────────  │
│  • Torso angle (θ₁): shoulder_mid → hip_mid            │
│  • Leg angle (θ₂): hip_mid → knee_mid/ankle_mid        │
│  • Knee angles: hip → knee → ankle (left & right)      │
│  • Hip height ratio: hip_y / body_height               │
│  • Segment alignment: θ₁ vs θ₂                         │
└─────────────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 2: Image Tilt Correction                         │
│  ─────────────────────────────────────────────────────  │
│  IF segments_aligned AND both non-vertical:            │
│    → Image is tilted                                    │
│    → Calculate tilt_offset                              │
│    → Correct all angles                                 │
└─────────────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 3: Priority Classification                       │
│  ─────────────────────────────────────────────────────  │
│  1️ LYING:                                              │
│     ✓ Torso horizontal AND legs horizontal             │
│                                                          │
│  2️ SITTING (Priority check - catches front-facing):   │
│     ✓ Knees bent (<120°) AND hip low (<0.50)           │
│     ✓ OR: 2+ indicators (hip ratio, knee bend, angles) │
│                                                          │
│  3️ STANDING:                                           │
│     ✓ Torso vertical + legs vertical + aligned         │
│     ✓ OR: Torso vertical + at least ONE knee straight  │
│                                                          │
│  4️ SQUATTING:                                          │
│     ✓ Knee angle in range (60°-120°)                   │
│     ✓ Hip at medium height (>0.45)                     │
│                                                          │
│  5️ BENDING:                                            │
│     ✓ Legs vertical + torso bent forward               │
│     ✓ Large angle difference (>20°)                    │
└─────────────────────────────────────────────────────────┘
```

### **Why This Works:**

**Camera-Angle Invariant Features:**
- **Knee angles** are consistent regardless of camera position:
  - Sitting (front or side): 90°-120°
  - Standing (front or side): 150°-180°
  
**Priority-Based Detection:**
- Checks **sitting BEFORE standing** to avoid false positives
- Front-facing sitting caught by: bent knees + low hip
- Standing with movement caught by: vertical torso + one straight knee

**Multi-Cue Voting:**
- No single measurement can fail the classification
- Each posture has 2-4 independent indicators
- Requires multiple indicators to agree

---

## 📊 Example Output
```
======================================================================
Result 0, Person 0
======================================================================
POSTURE: Sitting (confidence: 0.85)

Geometric Features:
  Torso angle: -88.3° (deviation from vertical: 1.7°)
  Leg angle: 92.1° (deviation from vertical: 2.1°)
  Average knee angle: 105.4°
  Hip height ratio: 0.42

Classification Logic:
  ✓ Knees bent (105.4° < 120°)
  ✓ Hip low (0.42 < 0.50)
  → Strong sitting signal detected
======================================================================
```

---

##  Getting Started

### **Installation**
```bash
# Clone repository
git clone <your-repo-url>
cd posture-detection-service

# Install dependencies
pip install ultralytics opencv-python numpy

# Download YOLO pose model (auto-downloaded on first run)
# Or manually: https://github.com/ultralytics/assets/releases
```

### **Quick Start**
```python
from service.posture_recognition_service import PoseRecognition

# For video file
service = PoseRecognition(
    video_source="path/to/video.mp4",
    source_id="video_001",
    model_path="yolo11m-pose.pt"
)
service.run()

# For live camera
service = PoseRecognition(
    video_source=0,  # Camera index
    source_id="camera_001",
    model_path="yolo11m-pose.pt"
)
service.run()

# For RTSP stream
service = PoseRecognition(
    video_source="rtsp://192.168.1.100:554/stream",
    source_id="rtsp_cam1",
    model_path="yolo11m-pose.pt"
)
service.run()
```

---

## 🎛️ Configuration & Tuning

### **Adjusting Detection Thresholds**

Edit `config/posture_config.py`:
```python
@dataclass
class PostureConfig:
    # Make sitting detection more sensitive
    sitting_knee_angle_max: float = 125.0  # Default: 120.0
    
    # Make standing detection more relaxed
    standing_knee_angle_min: float = 145.0  # Default: 150.0
    vertical_deviation_threshold: float = 15.0  # Default: 12.0
    
    # Adjust hip height thresholds
    sitting_hip_height_ratio_max: float = 0.48  # Default: 0.45
```

### **Common Tuning Scenarios**

**Too many "Unknown" classifications:**
```python
vertical_deviation_threshold: float = 15.0  # More tolerant
```

**Sitting not detected for front-facing people:**
```python
sitting_knee_angle_max: float = 130.0  # Higher threshold
sitting_hip_height_ratio_max: float = 0.50  # More lenient
```

**Standing misclassified as sitting:**
```python
standing_knee_angle_min: float = 145.0  # Lower threshold
```

---

##  Project Structure
```
posture-detection-service/
├── config/
│   └── posture_config.py          # All detection thresholds
├── core/
│   ├── classifier.py              # Main classification logic
│   └── geometry.py                # Geometric calculations
├── preprocessing/
│   └── keypoint_extractor.py      # Keypoint extraction & validation
├── service/
│   ├── posture_recognition_service.py  # Main service orchestrator
│   └── frame_annotator.py         # Visualization
├── stream/
│   └── stream_handler.py          # Video stream management
├── utils/
│   └── logger.py                  # Logging utilities
└── main.py                        # Entry point
```

---

## Performance Considerations

- **Motion detection**: Reduces compute by 60-80% (only processes moving frames)
- **Multi-stream**: Each stream runs in separate process/thread

---


**Built with using YOLO, OpenCV, and geometric reasoning**