# attendance-system-using-face-detection
This project demonstrates a simple face recognition system using Python and OpenCV. The system captures faces through a webcam, recognizes them using an Eigenfaces model, and logs the recognized faces with their details in an Excel file. It provides real-time feedback by displaying recognized faces and their details on the screen.

## Features

- **Face Detection**: Utilizes OpenCV's Haar Cascade classifier to detect faces in live webcam feed.
- **Face Recognition**: Implements an Eigenfaces model from OpenCV to recognize faces based on pre-trained data.
- **Excel Logging**: Logs recognized faces with their ID, name, university, and timestamp into an Excel file.
- **Real-time Feedback**: Displays recognized faces and their details (ID, name, university) on the screen.

## Setup Instructions

1. **Clone the Repository:**
   ```bash
   https://github.com/cadmostafijur/attendance-system-using-face-detection.git
   cd attendance-system-using-face-detection.git
   pip install opencv-python numpy openpyxl
   python check_face.py
   python face.py

