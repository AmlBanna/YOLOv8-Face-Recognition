# 🧠 YOLOv8 Face Recognition

## 📖 Overview

This project implements a **Face Recognition System** powered by **YOLOv8**.
It combines **face detection** and **face classification** to recognize celebrity identities in real-time from images or videos.
The system is trained on a **celebrity face dataset** and fine-tuned using **Ultralytics YOLOv8 classification** mode for accurate and efficient recognition.

---

## 🎥 Demo
[![Demo Video](Demo/Tsst.gif)](Demo/Video_Testing.mp4)

## 🚀 Key Features

* 🧩 **Face Detection** using YOLOv8 for precise localization.
* 🧠 **Face Classification** with YOLOv8-cls to recognize individual identities.
* ⚙️ **Automatic Data Preprocessing** — cleaning, balancing, and augmenting classes.
* 📊 **Exploratory Data Analysis (EDA)** for class distribution and dataset insight.
* 🎥 **Video Inference Pipeline** — real-time face recognition on live streams or saved videos.
* 🌐 **Gradio Interface** for easy testing via web UI.

---


## 🧰 Tech Stack

| Category               | Tools / Libraries                                                |
| ---------------------- | ---------------------------------------------------------------- |
| Framework              | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) |
| Programming            | Python                                                           |
| Data Handling          | Pandas, NumPy                                                    |
| Visualization          | Matplotlib, Seaborn                                              |
| Computer Vision        | OpenCV                                                           |
| Web Interface          | Gradio                                                           |
| Machine Learning Utils | Scikit-learn, Imbalanced-learn                                   |

---

## 📦 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/YOLOv8_Face_Recognition.git
cd YOLOv8_Face_Recognition
```

### 2️⃣ Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate       # On macOS / Linux
venv\Scripts\activate          # On Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Dataset Preparation

1. Download the **Celebrity Face Recognition Dataset** from Kaggle:
   [https://www.kaggle.com/datasets/bhaveshmittal/celebrity-face-recognition-dataset](https://www.kaggle.com/datasets/bhaveshmittal/celebrity-face-recognition-dataset)

2. Place your `kaggle.json` file in the appropriate directory:

   ```bash
   mkdir ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. Run the notebook cell that downloads and unzips the dataset automatically.

4. The notebook will clean, balance, and split data into **train / val / test** sets in YOLO format.

---

## ⚙️ Training the Model

In the notebook, configure your training parameters such as:

```python
model = YOLO("yolov8s-cls.pt")
model.train(data="data", epochs=50, imgsz=224, batch=32, optimizer="Adam")
```

* **`yolov8s-cls.pt`** is a pretrained YOLOv8 classification model.
* The dataset structure automatically matches YOLOv8’s classification input format.

During training, you’ll see metrics like accuracy, loss curves, and confusion matrices.

---

## 🎬 Inference Pipeline

After training:

1. Load your trained model and a YOLOv8 face detector.
2. Detect faces in frames using:

   ```python
   results = detector.predict(source=frame)
   ```
3. Crop detected faces and classify each:

   ```python
   pred = classifier.predict(source=face_crop)
   ```
4. Annotate frames with bounding boxes, labels, and confidence scores.
5. Save or stream the processed video with:

   ```python
   cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
   ```

---

## 🌐 Gradio Web App (Optional)

The notebook includes a Gradio demo interface where users can:

* Upload an image or video
* See recognized faces and confidence levels directly in the browser

To launch:

```python
gr.Interface(fn=inference_function, inputs="image", outputs="image").launch()
```

--- 
