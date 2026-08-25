# Hand Shape Hologram

An interactive computer vision project that tracks both hands via webcam and renders a dynamic, holographic-colored polygon between them, controllable through hand movement and gesture drawing.

## How It Works

Detects both hands in real time using MediaPipe Hand Landmarker. Draws a polygon anchored between the two palm centers. Distance between hands controls the polygon size. Angle between hands controls the polygon rotation. The shape can be a random irregular polygon, or a custom shape drawn live with your index finger. Inside the shape, a holographic color effect blends with the real camera background, with optional overlay patterns.

## Requirements

Python 3.9 or newer. A working webcam.

## Installation

git clone https://github.com/kamandNajari/Hand_Shape_Hologram.git
cd Hand_Shape_Hologram
pip install -r requirements.txt

The hand tracking model file is downloaded automatically on first run, no manual setup needed.

## Usage

python hand_shape_hologram.py

Show both hands to the camera to see the polygon appear between them.

## Controls

Key d: Start drawing mode, trace a custom shape in the air with your index finger
Key f: Finish drawing, the traced shape replaces the current polygon
Key r: Generate a new random irregular polygon
Key p: Cycle through visual modes, hologram only, then colorful checkerboard, then animated random shapes
Key q: Quit the application

## Tech Stack

OpenCV for video capture and rendering. MediaPipe for hand landmark detection. NumPy for numerical operations.

## License

MIT

