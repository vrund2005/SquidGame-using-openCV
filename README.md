# 🦑 Squid Game - Red Light Green Light (OpenCV + MediaPipe)

A fun computer vision game inspired by **Squid Game’s Red Light, Green Light** challenge.  
The player must stay still when the **Red Light** is on and can move only during the **Green Light**.  
If movement is detected during the Red Light, the player is **eliminated**!  

---

## 🎮 Features
- Uses **OpenCV** for video processing.  
- **MediaPipe Pose Estimation** to detect body keypoints.  
- Doll images displayed on screen (Red Light / Green Light).  
- Random light switches just like in the real Squid Game.  
- Movement detection using **nose + leg keypoints**.  

---

## ⚙️ Requirements
- Python 3.9+  
- Libraries:
  - `opencv-python`
  - `mediapipe`
  - `numpy`

Install dependencies:
```bash
pip install opencv-python mediapipe numpy
```

## 📜 License
- This project is for educational & fun purposes only.
Not affiliated with Netflix or Squid Game.
## ✨ Author
Developed by Patel Vrund Kalpeshbhai
