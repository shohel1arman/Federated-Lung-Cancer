# Federated Learning Framework for Lung Cancer Detection

A privacy-preserving **Federated Learning (FL)** system designed for the classification of Lung CT scans. Built with **PyTorch** and **Flower (flwr)**, this project integrates **Explainable AI (XAI)** metrics directly into the validation loop to ensure model reliability and faithfulness across distributed clients.

## 🚀 Key Features

* **Federated Architecture**: Uses a custom `MedicalFLStrategy` (based on FedAvg) to handle dynamic configuration broadcasting, global model checkpointing, and history tracking.
* **Explainable AI (XAI) Probe**: Clients automatically compute **Grad-CAM++** heatmaps and **Deletion AUC** scores during validation. This quantifies how "faithful" the model's explanations are without sharing patient data.
* **Medical-Grade Preprocessing**:
    * **CLAHE** (Contrast Limited Adaptive Histogram Equalization) for enhancing CT scan structures.
    * **Albumentations** for robust data augmentation (Shift, Scale, Rotate, Flip).
* **Advanced Training Pipeline**:
    * **Loss Functions**: Support for CrossEntropy, **Focal Loss** (for class imbalance), and **Label Smoothing**.
    * **Schedulers**: ReduceLROnPlateau, Cosine Annealing, and StepLR.
    * **Metrics**: Tracks Sensitivity, Specificity, F1-Macro, and AUC-ROC.
* **Multi-Model Support**: Easily switch between architectures: `customcnn`, `resnet50`, `densenet121`, and `mobilenetv3`.

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `server.py` | Central FL server. Orchestrates rounds, aggregates weights, plots training curves, and saves the global model. |
| `client.py` | FL client. Handles local training, validation, and XAI probing. Supports standalone (non-federated) training. |
| `dataloder.py` | Manages `CTScanDataset`. Handles image loading, CLAHE preprocessing, and splitting (Train/Val/Test). |
| `train_eval.py` | Contains the `ModelTrainer` engine, `ModelMetrics` calculation, and TensorBoard logging logic. |
| `model_factory.py` | Factory pattern for instantiating models (ResNet, DenseNet, etc.) and custom loss functions. |

## 🛠️ Installation

1.  **Clone the repository**
    ```bash
    git clone <repository-url>
    cd lung-cancer-fl
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/macOS
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install torch torchvision numpy pandas scikit-learn opencv-python albumentations flwr matplotlib seaborn tqdm tensorboard
    ```

## 📊 Dataset Setup

The system expects data to be organized by class folders. The `dataloder.py` script automatically detects classes based on folder names.

```text
/DataSet/Lung-CT-Scan/
├── Bengin cases/
│   ├── image_01.jpg
│   └── ...
├── Malignant cases/
│   ├── image_01.jpg
│   └── ...
└── Normal cases/
    ├── image_01.jpg
    └── ...
```
🚀 Usage
### 1. Start the Server
Run the server first. It will wait for the specified number of clients to connect.

````bash
python server.py --rounds 10 --min-clients 2 --model customcnn
````
--rounds: Number of federated learning rounds.

--min-clients: Minimum clients required to start a round.

--model: Architecture to use (customcnn, resnet50, densenet121, mobilenetv3).

2. Start Clients
Open separate terminals for each client. Ensure client-id is unique.

### Client 1
````bash
python client.py --client-id 1 --data-dir "path/to/dataset" --server-address "localhost:8080"
````

### Client 2
````bash
python client.py --client-id 2 --data-dir "path/to/dataset" --server-address "localhost:8080"
````

📈 Outputs & Results
Results are saved in the Result directory, organized by session timestamp.

Result/FLResult/ (Server Outputs):

training_curves.png: Plots for Loss, Accuracy, F1, and Client counts.

best_model_round_X.pth: Checkpoint of the best performing global model.

history_round_X.json: Detailed training logs.

Result/clientresult/ (Client Outputs):

xai/: Contains Grad-CAM++ overlays with Deletion AUC scores.

metrics/: Confusion matrices and classification reports.

checkpoints/: Local best model weights.

🧠 Explainable AI (XAI) Details
The client performs a "faithfulness probe" on the validation set during evaluation:

Grad-CAM++: Generates a heatmap highlighting important regions.

Deletion AUC: Progressively removes high-activation pixels and measures the drop in model confidence.

Lower Score (< 0.3): High faithfulness (the model is truly looking at the tumor).

Higher Score (> 0.7): Low faithfulness (the model might be looking at background noise).