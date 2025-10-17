🕺 **Dance Recognition Project — Interactive ML Learning Tool**

📘 The Dance Recognition Project is an educational machine learning tool that teaches computer vision, pose estimation, and sequence modeling through an interactive dance recognition system.

Using a Raspberry Pi 4 and Pi Camera V2, the system captures video, extracts pose keypoints via MediaPipe, and classifies dance moves using a TensorFlow LSTM model. It helps learners understand how data collection, model training, and real-time inference work together to recognize motion.

🎯 Features

*   Real-time pose tracking with MediaPipe
    
*   Sequence modeling using an LSTM network
    
*   Modular design: separate scripts for data, training, and testing
    
*   Optimized for Raspberry Pi
    
*   Interactive and educational for beginners
    

🧠 System OverviewCamera Feed → MediaPipe Pose → Keypoints → LSTM Model → Predicted Move

Files:

*   new\_moves.py — Collect and save pose data
    
*   move\_train.py — Train LSTM model on recorded data
    
*   live\_dance.py — Run live predictions and display results
    

⚙️ Setup

Requirements:Python 3.9+, TensorFlow 2.x, MediaPipe, OpenCV, NumPy, Pandas

Installation:

1.  git clone [https://github.com/yourusername/dance-recognition.git](https://github.com/yourusername/dance-recognition.git)
    
2.  cd dance-recognition
    
3.  pip install -r requirements.txt
    

Enable Pi Camera via raspi-config and position it for full-body capture.

📊 Data CollectionCommand: python3 new\_moves.pyPerform each move as prompted. Keypoints are saved as .npy datasets for training.

🏋️ Model Training Command: python3 move\_train.py Trains an LSTM model on saved data and outputs model.h5.You can adjust sequence length, hidden units, and epochs for experimentation.

🎥 Live RecognitionCommand: python3 live\_dance.pyRuns real-time pose tracking and displays predicted moves over the camera feed.

🧩 Learning Goals

*   Understand how pose data encodes motion
    
*   Learn temporal modeling with RNNs/LSTMs
    
*   Explore inference and latency on edge devices
    

🚀 Future Enhancements

*   Convert to TensorFlow Lite for faster performance
    
*   Add a GUI interface
    
*   Expand the dataset and add move accuracy feedback
    
*   Gamification features for classroom learning
    

🧑‍💻 Structuredance-recognition/├── new\_moves.py├── move\_train.py├── live\_dance.py├── model.h5├── data/└── README.md

👤 AuthorDonovan ThomasComputer Science Student — University of California, Santa CruzEducational ML Project (Pose Recognition & Interactive Learning)

📚 Acknowledgments

*   MediaPipe — Pose estimation
    
*   TensorFlow — Deep learning
    
*   OpenCV — Video processing
    
*   UCSC Scharf Lab — Research inspiration
    

🧩 Requirementstensorflow==2.14.0mediapipe==0.10.8opencv-python==4.10.0.84numpy==1.26.4pandas==2.2.2matplotlib==3.9.2scikit-learn==1.5.2jupyterlab==4.2.5tflite-runtime; platform\_machine=="armv7l"
