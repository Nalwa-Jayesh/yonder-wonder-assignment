# Age Classification Deep Learning Project

A comprehensive deep learning solution for age classification from facial images using PyTorch and computer vision techniques. This project implements a hybrid CNN architecture that combines both regression and classification approaches for accurate age prediction.

## 🎯 Project Overview

This project develops an age classification model that can predict a person's age from facial images across different age groups (20s, 30s, 40s, 50s, 60s, 70s, 80s). The model uses a sophisticated dual-head architecture that leverages both regression and classification techniques to achieve superior accuracy.

### Key Features

- **Hybrid Architecture**: Combines regression and classification heads for robust age prediction
- **Transfer Learning**: Utilizes pre-trained ResNet18 backbone for improved feature extraction
- **Advanced Data Augmentation**: Implements strong augmentation techniques to improve generalization
- **Test Time Augmentation (TTA)**: Enhances prediction accuracy through ensemble predictions
- **Stratified Data Splitting**: Ensures balanced training and validation sets
- **Comprehensive Evaluation**: Detailed per-age-group analysis and visualization tools
- **Early Stopping**: Prevents overfitting with patience-based training termination

## 🏗️ Architecture

The model architecture consists of:

1. **Backbone**: Pre-trained ResNet18 (with ImageNet pre-trained weights) or custom CNN
2. **Feature Processor**: Fully connected layers with dropout for regularization
3. **Dual Heads**:
   - **Regression Head**: Predicts continuous age values
   - **Classification Head**: Classifies into age groups (7 classes)
4. **Combined Loss**: Weighted combination of regression (MAE/MSE) and classification (CrossEntropy) losses

## 📊 Dataset Structure

```
data/assessment-data/
├── 20/          # Images of people in their 20s
├── 30/          # Images of people in their 30s
├── 40/          # Images of people in their 40s
├── 50/          # Images of people in their 50s
├── 60/          # Images of people in their 60s
├── 70/          # Images of people in their 70s
└── 80/          # Images of people in their 80s
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM (16GB+ recommended for optimal performance)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd yonder-ai
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # or
   .venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your data**:
   - Extract the `assessment_exercise_data.zip` file
   - Ensure images are organized in the folder structure shown above
   - Place the data in `data/assessment-data/`

### Usage

#### Training the Model

Run the complete training pipeline:

```bash
python main.py
```

This will:
- Load and preprocess the dataset
- Split data into training/validation sets (70/30 split)
- Apply data augmentation and create augmented training samples
- Train the model with early stopping
- Evaluate performance with Test Time Augmentation
- Save the trained model and generate visualizations

#### Configuration Options

Key parameters can be modified in `main.py`:

```python
# Training Configuration
BATCH_SIZE = 8                    # Batch size (can increase to 16-32 with 16GB+ RAM)
LEARNING_RATE = 0.0005           # Learning rate
NUM_EPOCHS = 150                 # Maximum epochs
PATIENCE = 20                    # Early stopping patience
TRAIN_RATIO = 0.7               # Training/validation split
AUGMENTATION_MULTIPLIER = 5      # Data augmentation factor (can increase to 8-10)
USE_STRONG_AUGMENTATION = True   # Enable advanced augmentations
USE_TTA = True                  # Enable Test Time Augmentation
```

## 📁 Project Structure

```
yonder-ai/
├── config/                 # Configuration files
│   ├── __init__.py
│   └── config.py          # Project configuration settings
├── data/                  # Data handling and preprocessing
│   ├── __init__.py
│   ├── dataset.py         # Custom PyTorch dataset classes
│   ├── transforms.py      # Data augmentation and preprocessing
│   ├── assessment-data/   # Training data directory
│   └── assessment_exercise_data.zip
├── models/                # Model definitions and saved weights
│   ├── __init__.py
│   ├── age_classifier.py  # Main CNN architecture
│   ├── losses.py         # Custom loss functions
│   ├── final_age_model.pth    # Final trained model
│   └── best_age_model.pth     # Best model checkpoint
├── training/              # Training and evaluation logic
│   ├── __init__.py
│   ├── trainer.py        # Training loop and model training
│   └── evaluator.py      # Model evaluation and metrics
├── utils/                 # Utility functions and helpers
│   ├── __init__.py
│   ├── visualization.py  # Plotting and visualization tools
│   └── helpers.py        # Helper functions and utilities
├── main.py               # Main training script
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🔧 Model Components

### Data Augmentation

The project implements comprehensive data augmentation:

- **Geometric**: Rotation, flipping, resizing
- **Color**: Brightness, contrast, saturation adjustments
- **Advanced**: Gaussian blur, random erasing, perspective transforms
- **Normalization**: ImageNet normalization statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Loss Function

Uses a combined loss approach:
- **α × Regression Loss** (MAE/MSE for continuous age prediction)
- **β × Classification Loss** (CrossEntropy for age group classification)
- Default weights: α=0.8, β=0.2

### Training Strategy

- **Differential Learning Rates**: Lower rates for pre-trained backbone, higher for new heads
- **Learning Rate Scheduling**: ReduceLROnPlateau with factor=0.5
- **Early Stopping**: Monitors validation loss with configurable patience
- **Gradient Clipping**: Prevents gradient explosion

## 📈 Performance Metrics

The model provides comprehensive evaluation metrics:

- **Mean Absolute Error (MAE)**: Primary regression metric
- **Root Mean Square Error (RMSE)**: Additional regression metric  
- **Per-Age-Group Analysis**: Detailed breakdown by age ranges
- **Training History**: Loss and accuracy curves over epochs
- **Prediction Visualizations**: Scatter plots of predicted vs. actual ages

## 🎨 Visualization Features

The project generates several visualizations:

1. **Data Distribution**: Histogram of age distribution in train/validation sets
2. **Training History**: Loss and metric curves over epochs
3. **Prediction Analysis**: Scatter plot of predictions vs. ground truth
4. **Age Group Performance**: Per-group accuracy breakdown

## ⚙️ Advanced Features

### Test Time Augmentation (TTA)
- Applies multiple augmentations during inference
- Averages predictions for improved accuracy
- Configurable number of TTA iterations

### Stratified Data Splitting
- Maintains age distribution balance across train/validation sets
- Ensures representative sampling for each age group

### Model Checkpointing
- Saves best model based on validation performance
- Automatic model persistence after training completion

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce `BATCH_SIZE` in main.py (try 16, then 8 if needed)
   - For systems with 16GB+ RAM, you can typically use larger batch sizes

2. **Data Loading Errors**:
   - Verify data directory structure matches expected format
   - Check image file extensions (.png, .jpg, .jpeg)

3. **Poor Performance**:
   - Increase `NUM_EPOCHS` or adjust `PATIENCE`
   - Experiment with different `LEARNING_RATE` values
   - Ensure sufficient training data
   - With ample RAM, try increasing `AUGMENTATION_MULTIPLIER` to 8-10

### Performance Tips

- Use GPU acceleration when available
- Increase `num_workers` in DataLoaders for faster data loading (try 4-8 with 16GB+ RAM)
- With high memory systems, increase `BATCH_SIZE` to 16-32 for faster training
- Experiment with higher `AUGMENTATION_MULTIPLIER` values (8-10) for better generalization
- Enable `pin_memory=True` for faster GPU transfers
- Consider ensemble methods for production use

## 📚 Dependencies

Key packages used in this project:

- **PyTorch**: Deep learning framework
- **torchvision**: Computer vision utilities
- **NumPy**: Numerical computing
- **matplotlib**: Visualization
- **scikit-learn**: Machine learning utilities
- **Pillow**: Image processing

See `requirements.txt` for complete dependency list with versions.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Pre-trained ResNet18 model from torchvision (using ImageNet pre-trained weights)
- ImageNet normalization statistics for optimal transfer learning performance
- PyTorch community for excellent documentation and tools

---

For questions or issues, please open an issue in the repository or contact the development team.