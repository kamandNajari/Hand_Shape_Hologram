# Hand Shape Hologram

An interactive computer vision project that tracks both hands via webcam and renders a dynamic, holographic-colored polygon between them — controllable through hand movement and live gesture drawing.

## ✨ Features

- Real-time tracking of both hands using **MediaPipe Hand Landmarker**
- A polygon rendered between the two palm centers
- **Distance** between hands controls the polygon's size
- **Angle** between hands controls the polygon's rotation
- Draw your own custom shape live with your index finger
- Holographic color effect blended with the real camera background
- Extra animated overlay patterns (checkerboard, shifting icons)

## 🎥 Demo

*(add a screenshot or GIF here)*

## 📦 Requirements

- Python 3.9 or newer
- A working webcam
- 
## 🚀 Installation

```bash
git clone https://github.com/kamandNajari/Hand_Shape_Hologram.git
cd Hand_Shape_Hologram
pip install -r requirements.txt
```

The hand tracking model is downloaded automatically on first run — no manual setup needed.

## ▶️ Usage

```bash
python hand_shape_hologram.py
```

Show both hands to the camera to see the polygon appear between them.


## 🎮 Controls

| Key | Action |
|-----|--------|
| `d` | Start drawing mode — trace a custom shape in the air with your index finger |
| `f` | Finish drawing — the traced shape replaces the current polygon |
| `r` | Generate a new random irregular polygon |
| `p` | Cycle visual modes — hologram only → colorful checkerboard → animated random shapes |
| `q` | Quit the application |

## 🛠️ Tech Stack

- [OpenCV](https://opencv.org/) — video capture and rendering
- [MediaPipe](https://developers.google.com/mediapipe) — hand landmark detection
- [NumPy](https://numpy.org/) — numerical operations for shape and color effects

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
