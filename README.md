## Image Classification using CNN (CIFAR & MNIST)

This project demonstrates image classification using **Convolutional Neural Networks (CNNs)** on two popular datasets: **CIFAR-10** and **MNIST**. It uses **TensorFlow/Keras** to build, train, evaluate, and use models to predict image classes.

---

## 📌 Features

- CNN models trained on CIFAR-10 and MNIST datasets
- Real-time visualization of training and validation metrics
- Image preprocessing and augmentation
- Save/load trained models
- Predict labels for unseen/test images

---

## 🧰 Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- PIL or OpenCV (for image processing)

---

## 🗂️ Project Structure

Image-Classification-Using-CNN/
│
├── models/ # Saved CNN model files
├── cnn_cifar.py # CNN training & evaluation on CIFAR-10
├── cnn_mnist.py # CNN training & evaluation on MNIST
├── predict.py # Prediction script for custom images
├── requirements.txt # Required Python packages
└── README.md # Project documentation

yaml
Copy
Edit

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/image-classification-using-cnn.git
   cd image-classification-using-cnn
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Train on CIFAR-10

1.bash
2.Copy
3.Edit
4.python cnn_cifar.py
5.Train on MNIST

bash
Copy
Edit
python cnn_mnist.py
Make predictions

bash
Copy
Edit
python predict.py --image path_to_image.jpg
📊 Output
Training/Validation Accuracy & Loss Graphs

Predicted class for a custom test image

Model architecture summary in terminal

✅ To Do
Add Streamlit or Flask web interface

Include confusion matrix and F1-score

Try transfer learning with VGG16 or ResNet

Extend to multi-class image datasets

🤝 Contributing
Contributions are welcome! Fork the repo and submit a pull request. For major changes, open an issue to discuss ideas.
