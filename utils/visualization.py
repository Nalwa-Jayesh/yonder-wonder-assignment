# Visualization utilities 
import matplotlib.pyplot as plt
import numpy as np

def plot_data_distribution(train_labels, val_labels):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(train_labels, bins=7, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Training Set Distribution (n={len(train_labels)})')
    axes[0].set_xticks([20, 30, 40, 50, 60, 70, 80])
    axes[0].grid(True, alpha=0.3)
    axes[1].hist(val_labels, bins=7, alpha=0.7, color='red', edgecolor='black')
    axes[1].set_xlabel('Age')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Validation Set Distribution (n={len(val_labels)})')
    axes[1].set_xticks([20, 30, 40, 50, 60, 70, 80])
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print("Data Distribution:")
    print("Training set:")
    for age in [20, 30, 40, 50, 60, 70, 80]:
        count = sum(1 for x in train_labels if x == age)
        print(f"  Age {age}: {count} samples")
    print("Validation set:")
    for age in [20, 30, 40, 50, 60, 70, 80]:
        count = sum(1 for x in val_labels if x == age)
        print(f"  Age {age}: {count} samples")

def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_losses'], label='Train Loss')
    axes[0].plot(history['val_losses'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(history['train_maes'], label='Train MAE')
    axes[1].plot(history['val_maes'], label='Val MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE (years)')
    axes[1].set_title('Training and Validation MAE')
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plt.show()

def plot_predictions(predictions, targets):
    plt.figure(figsize=(10, 8))
    plt.scatter(targets, predictions, alpha=0.6)
    plt.plot([20, 80], [20, 80], 'r--', label='Perfect Prediction')
    plt.xlabel('True Age')
    plt.ylabel('Predicted Age')
    plt.title('Age Predictions vs True Ages')
    plt.legend()
    plt.grid(True)
    plt.show() 