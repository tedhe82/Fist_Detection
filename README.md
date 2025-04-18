# Fist Detection using MediaPipe

This project uses MediaPipe and OpenCV to detect fists in real-time using a webcam. It includes both a Python script and a Jupyter notebook implementation.

## Requirements
- Python 3.7+
- Webcam
- Required packages:
  - mediapipe
  - opencv-python
  - numpy
  - ipywidgets (for Jupyter notebook version)

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
pip install ipywidgets  # For Jupyter notebook version
```

## Usage

### Python Script Version
Run the application:
```bash
python fist_detection.py
```

The application will:
1. Open your webcam
2. Detect hands in real-time
3. Draw landmarks on detected hands
4. Draw a green rectangle and label when a fist is detected
5. Count and display the number of fists detected
6. Press 'r' to reset the counter
7. Press 'q' to quit the application

### Jupyter Notebook Version
1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `fist_detection.ipynb`

3. Run the cells in order:
   - First cell: Imports required packages
   - Second cell: Defines the FistDetector class
   - Third cell: Creates the interactive buttons
   - Fourth cell: Sets up the detection functionality

4. Use the interactive buttons to:
   - Start Detection: Starts the webcam and fist detection
   - Stop Detection: Stops the webcam and detection
   - Reset Count: Resets the fist counter to zero

## Features

- Real-time fist detection using MediaPipe
- Hand landmark visualization
- Fist counting functionality
- Interactive controls (in notebook version)
- Counter reset capability
- Webcam feed display

## How it Works

The application uses MediaPipe's hand tracking solution to detect hands and their landmarks. It then calculates the distance between the thumb and index finger to determine if a fist is being made. When the distance is small enough, it's considered a fist and highlighted on the screen.

The counter increments each time a new fist is detected (not continuously while holding a fist), and can be reset using either the 'r' key (script version) or the Reset Count button (notebook version).