# Automatic Number Plate Recognition (ANPR) with YOLOv9 and EasyOCR

## 📋 Overview
This project implements an Automatic Number Plate Recognition (ANPR) system using YOLOv9 for license plate detection and EasyOCR for text extraction from the detected plates. The model is trained on a custom dataset from Roboflow Universe.

## 🚀 Features
- License plate detection using YOLOv9
- Text extraction from detected plates using EasyOCR
- GPU-accelerated training and inference
- Support for custom datasets from Roboflow
- Real-time plate recognition capabilities

## 📁 Project Structure
```
├── README.md
├── requirements.txt
├── train.py
├── detect_plate.py
├── anpr_pipeline.py
└── notebooks/
    └── anpr_yolov9_training.ipynb
```

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional but recommended)
- Google Colab (for training) or local machine with GPU

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/anpr-yolov9.git
cd anpr-yolov9
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Clone YOLOv9 repository**
```bash
git clone https://github.com/SkalskiP/yolov9.git
cd yolov9
pip install -r requirements.txt
cd ..
```

4. **Download pre-trained weights**
```bash
mkdir weights
wget -P weights https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c.pt
wget -P weights https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt
```

## 📊 Dataset

The project uses the ANPR2 dataset from Roboflow Universe:
- **Dataset**: [ANPR2 - License Plate Detection](https://universe.roboflow.com/arvind-kumar-wjygd/anpr2-syxl7)
- **Format**: YOLOv9 compatible format
- **Classes**: 1 (license-plate)

### Download Dataset
```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("arvind-kumar-wjygd").project("anpr2-syxl7")
version = project.version(8)
dataset = version.download("yolov9")
```

## 🏋️ Training

### In Google Colab
Open the provided notebook `notebooks/anpr_yolov9_training.ipynb` in Google Colab and follow the steps.

### Local Training
```bash
cd yolov9

python train.py \
--batch 16 \
--epochs 25 \
--img 640 \
--device 0 \
--data /path/to/dataset/data.yaml \
--weights ../weights/gelan-c.pt \
--cfg models/detect/gelan-c.yaml \
--hyp hyp.scratch-high.yaml
```

### Training Parameters
- `--batch`: Batch size (default: 16)
- `--epochs`: Number of training epochs (default: 25)
- `--img`: Image size (default: 640)
- `--device`: GPU device ID (use 'cpu' for CPU training)
- `--data`: Path to data.yaml file
- `--weights`: Pre-trained weights path
- `--cfg`: Model configuration file

## 🔍 Inference

### Single Image Detection
```python
from anpr_pipeline import ANPRPipeline

# Initialize the pipeline
anpr = ANPRPipeline(
    model_path='runs/train/exp/weights/best.pt',
    ocr_lang='en'
)

# Detect and read plate
result = anpr.process_image('path/to/image.jpg')
print(f"Plate Text: {result['text']}")
print(f"Confidence: {result['confidence']}")
```

### Batch Processing
```python
# Process multiple images
results = anpr.process_batch(['img1.jpg', 'img2.jpg', 'img3.jpg'])
for result in results:
    print(f"Image: {result['image']}, Plate: {result['text']}")
```

### Real-time Video Processing
```python
# Process video file or camera stream
anpr.process_video('input_video.mp4', 'output_video.mp4')
```

## 📦 Requirements

```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.5.0
easyocr>=1.6.0
roboflow>=1.0.0
matplotlib>=3.3.0
numpy>=1.21.0
pyyaml>=5.4.0
pandas>=1.3.0
tqdm>=4.64.0
pillow>=9.0.0
```

## 🚦 Usage Examples

### Complete ANPR Pipeline
```python
import cv2
from anpr_pipeline import ANPRPipeline

# Initialize ANPR system
anpr = ANPRPipeline(
    detection_model='path/to/best.pt',
    ocr_lang='en',
    gpu=True
)

# Load image
image = cv2.imread('car.jpg')

# Process image
result = anpr.process_image(image)

# Display results
if result['plate_detected']:
    print(f"License Plate: {result['text']}")
    print(f"Detection Confidence: {result['detection_confidence']:.2f}")
    print(f"OCR Confidence: {result['ocr_confidence']:.2f}")
    
    # Draw bounding box and text
    output_image = anpr.draw_results(image, result)
    cv2.imshow('ANPR Result', output_image)
    cv2.waitKey(0)
else:
    print("No license plate detected")
```

### Training from Scratch
```python
# Configure and train custom model
import subprocess

subprocess.run([
    "python", "train.py",
    "--batch", "16",
    "--epochs", "50",
    "--img", "640",
    "--data", "ANPR2-8/data.yaml",
    "--weights", "weights/gelan-c.pt",
    "--device", "0"
])
```

## 📈 Performance Metrics

After training for 25 epochs, the model achieves:
- **mAP@0.5**: ~95%
- **mAP@0.5:0.95**: ~75%
- **Precision**: ~90%
- **Recall**: ~88%

## 🛠️ Troubleshooting

### Common Issues and Solutions

1. **Dataset not found error**
   - Ensure paths in data.yaml are correct
   - Use absolute paths or proper relative paths
   ```python
   # Fix data.yaml paths
   data['train'] = '/absolute/path/to/train/images'
   data['val'] = '/absolute/path/to/valid/images'
   ```

2. **PyTorch compatibility issues**
   ```bash
   pip install torch==2.0.1 torchvision==0.15.2
   ```

3. **CUDA out of memory**
   - Reduce batch size
   - Use gradient accumulation
   ```bash
   python train.py --batch 8 --accumulate 2
   ```

4. **EasyOCR installation issues**
   ```bash
   pip install easyocr==1.6.0
   # If issues persist, install additional dependencies
   pip install opencv-python-headless
   ```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [YOLOv9](https://github.com/WongKinYiu/yolov9) by WongKinYiu
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) by JaidedAI
- [Roboflow](https://roboflow.com/) for the dataset and tools
- [SkalskiP](https://github.com/SkalskiP) for YOLOv9 fork with patches

## 📧 Contact

For questions or support, please open an issue on GitHub or contact:
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)

## 🚀 Future Improvements

- [ ] Add support for multiple languages in OCR
- [ ] Implement plate tracking across video frames
- [ ] Optimize for mobile deployment
- [ ] Add REST API endpoint
- [ ] Support for different plate formats (US, EU, Asian)
- [ ] Real-time video stream processing

## 🤝 Contributing

Contributions are welcome! Please read the contributing guidelines before submitting pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request