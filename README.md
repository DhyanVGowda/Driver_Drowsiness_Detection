# Driver Drowsiness Detection System

A Python-based project that leverages computer vision and machine learning to detect driver drowsiness in real-time, ensuring road safety. The system uses facial landmarks to monitor the driver's eye aspect ratio (EAR) and triggers alerts if drowsiness is detected.

## Features

- **Real-time Monitoring:** Detects drowsiness using a webcam feed.
- **Eye Aspect Ratio Calculation:** Monitors eye movements and blinks using facial landmarks.
- **Alert System:** Notifies the driver when drowsiness is detected.
- **User-Friendly Interface:** Simple and effective implementation for real-world applications.

## Project Inspiration

This project is inspired by other Driver Drowsiness Detection Systems from GitHub and the internet, with enhancements and customizations for better performance and usability.

## Technologies Used

- Python
- OpenCV
- Dlib
- Imutils
- Numpy

## Getting Started

### Prerequisites

Ensure you have Python installed on your system. You also need to install the following Python libraries:

- OpenCV
- Dlib
- Imutils
- Numpy

You can install them using the following command:

```bash
pip install opencv-python dlib imutils numpy

```

### Clone the Repository

Clone the project repository from GitHub:

```bash
git clone https://github.com/DhyanVGowda/Driver_Drowsiness_Detection.git
cd Driver_Drowsiness_Detection
```

### Usage

1. Run the `drowsiness_detection.py` script:
   ```bash
   python drowsiness_detection.py
   ```

### Usage

2. Allow access to your webcam for real-time monitoring.
3. The system will monitor the driver’s eyes and trigger an alert if drowsiness is detected.

### How It Works

1. **Facial Landmark Detection:** The system detects key facial landmarks using the Dlib library.
2. **Eye Aspect Ratio (EAR):** EAR is calculated for both eyes. If the ratio drops below a certain threshold for a continuous period, drowsiness is detected.
3. **Alert Mechanism:** An audible alert is triggered to notify the driver.

### Project Structure

```
Driver_Drowsiness_Detection/
│
├── drowsiness_detection.py   # Main script
├── README.md                 # Project documentation
└── requirements.txt          # List of dependencies
```

### Future Enhancements

- Add support for additional facial features to enhance detection accuracy.
- Integrate the system with IoT devices for a complete in-car solution.
- Optimize the system for mobile or embedded platforms.

### Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests.

### License

This project is open-source and available under the MIT License.
