# AI Hand Gesture Recognition

A real-time webcam-based hand gesture recognition project built with Python, OpenCV and MediaPipe. It detects a hand, tracks its 21 landmarks, and classifies four basic gestures.

## Supported gestures

- Open Hand
- Fist
- Thumbs Up
- Victory / Peace

## Technology

- **Python** – application logic
- **OpenCV** – webcam capture, image processing and display
- **MediaPipe Hands** – real-time hand landmark detection (21 landmarks)
- **NumPy** – numerical dependency used by the vision stack

This starter project uses MediaPipe's pre-trained hand landmark solution and rule-based landmark classification. It does not train a custom machine-learning model.

## Requirements

- Windows 10/11, macOS or Linux
- Python 3.10 or 3.11 recommended
- Working webcam
- Internet connection for the first dependency installation

## Setup (Windows / VS Code)

Open this folder in VS Code, then open Terminal.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `python` is not recognized, try `py` in the commands above.

## Run

```bash
python main.py
```

Allow camera access if your operating system asks. Hold one hand clearly in front of the camera. Press **Q** to close the application.

## Tips for better detection

- Keep your full hand visible inside the camera frame.
- Use a well-lit area and a plain background.
- Hold gestures upright and face your palm toward the camera.
- Move slowly while changing gestures.
- The simple rule-based classifier may not identify every hand orientation perfectly.

## Project structure

```text
AI_Hand_Gesture_Recognition/
├── main.py
├── requirements.txt
├── README.md
├── screenshots/
└── demo/
```

## Deliverables checklist

- [ ] Run the application and test all four gestures.
- [ ] Save screenshots of the running webcam window in `screenshots/`.
- [ ] Record a short demo and save it in `demo/`.
- [ ] Push the project to a public GitHub repository.

## Suggested demo

1. Start the application.
2. Show an open hand.
3. Make a fist.
4. Show thumbs up.
5. Show the victory/peace sign.
6. End by pressing Q.

## Notes

Gesture recognition is based on landmark positions and simple geometric rules. Performance depends on lighting, camera quality, hand orientation and the installed MediaPipe/OpenCV versions.
---

## 📸 Project Screenshots

![Open Hand](screenshots/open-hand.png)

![Fist](screenshots/fist.png)

![Thumbs Up](screenshots/thumbs-up.png)

![Victory](screenshots/victory.png)

![Two Hands](screenshots/two-hands.png)