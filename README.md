**🌳 Forest Animal Detection & Species Info**

🧠 YOLOv3 + OpenCV + Streamlit | Real-time Wildlife Monitoring

A smart AI system that detects forest animals from images & videos and shows useful species information like diet, habitat, scientific name, and conservation status.
Built using YOLOv3, OpenCV, and Streamlit Cloud 🚀

📌 Features
Feature	Description
🖼 Upload Image -	Detects each animal with bounding box
🎥 Upload Video -	Frame-by-frame detection with counting
🧠 AI Species Info Panel - Shows scientific info and fun facts
🔢 Animal Count	- Displays count of each species
🦌 Deer Fix -	Converts deer misclassification from “cow”
💾 Download Output -	Save annotated image/video
🌐 Live Deployment -	Accessible anywhere with link. 

🏛️ Project Architecture
 User Upload (Image/Video)
            ↓
       Pre-processing (OpenCV)
            ↓
 YOLOv3 Object Detection (COCO-trained)
            ↓
 Filter using Confidence + NMS
            ↓
 Bounding Boxes + Species Labels + Counts
            ↓
 Species Info Retrieval (Dictionary)
            ↓
 Final Output (Streamlit UI + Download)

🧠 Why YOLOv3?
Fastest object detector -	Real-time wildlife videos
Pre-trained model -	No training needed
Good accuracy -	Animals detected clearly
OpenCV support -	Easy integration

📌 YOLOv3 is trained on COCO dataset (80 classes), including
elephant, bear, bird, horse, zebra, giraffe, sheep, cow…

⚠ Limitation
COCO dataset does not include Deer, so deer often predicted as cow → solved using Deer Override.

🛠 Tech Stack
Category -	Tools
Code -	Python
AI Model -	YOLOv3 (Darknet-53)
Libraries -	OpenCV, NumPy, gdown
Deployment -	Streamlit Cloud
UI -	Streamlit

📂 Folder Structure
📁 Forest-Animal-Detection
│
├── app.py                              # Streamlit Web App
├── forest_animal_detectorl.py          # Core YOLO Detection Logic
├── yolov3.cfg                          # YOLO Model Architecture
├── coco.names                          # COCO Class Names
├── requirements.txt                    # Python Dependencies
│
├── sample_image.jpg (optional)
├── sample_video.mp4 (optional)
│
└── README.md                           # Project Documentation


⚠ yolov3.weights file is downloaded automatically on cloud using gdown

🌐 Live App (Try it Yourself!)

🔗 Hosted on Streamlit Cloud
➡ https://forest-detector.streamlit.app/


📌 First load may take time as weights are downloaded.

