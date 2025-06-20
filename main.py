import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from data.dataset import AgeDataset, create_augmented_dataset
from data.transforms import get_transforms
from models.age_classifier import AgeClassifierCNN
from models.losses import CombinedLoss
from training.trainer import train_model
from training.evaluator import evaluate_model, evaluate_model_with_tta
from utils.visualization import plot_data_distribution, plot_training_history, plot_predictions
from utils.helpers import stratified_split

def main():
    # Configuration
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0005
    NUM_EPOCHS = 150
    PATIENCE = 20
    DATA_DIR = 'data/assessment-data'
    TRAIN_RATIO = 0.7
    AUGMENTATION_MULTIPLIER = 8
    USE_STRONG_AUGMENTATION = True
    USE_TTA = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    base_dataset = AgeDataset(DATA_DIR, transform=None, regression=True)
    train_indices, val_indices = stratified_split(base_dataset, train_ratio=TRAIN_RATIO)
    print(f"\nDataset Split:")
    print(f"Total samples: {len(base_dataset)}")
    print(f"Training samples: {len(train_indices)} ({len(train_indices)/len(base_dataset)*100:.1f}%)")
    print(f"Validation samples: {len(val_indices)} ({len(val_indices)/len(base_dataset)*100:.1f}%)")

    train_transform = get_transforms(augment=True, strong_augment=USE_STRONG_AUGMENTATION)
    val_transform = get_transforms(augment=False)

    train_dataset = AgeDataset(DATA_DIR, transform=train_transform, regression=True)
    train_dataset.images = [train_dataset.images[i] for i in train_indices]
    train_dataset.labels = [train_dataset.labels[i] for i in train_indices]
    if AUGMENTATION_MULTIPLIER > 1:
        train_dataset = create_augmented_dataset(train_dataset, AUGMENTATION_MULTIPLIER)
    val_dataset = AgeDataset(DATA_DIR, transform=val_transform, regression=True)
    val_dataset.images = [val_dataset.images[i] for i in val_indices]
    val_dataset.labels = [val_dataset.labels[i] for i in val_indices]

    plot_data_distribution([base_dataset.labels[i] for i in train_indices],
                          [base_dataset.labels[i] for i in val_indices])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=6, pin_memory=True if device.type=='cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=6, pin_memory=True if device.type=='cuda' else False)

    print(f"\nFinal dataset sizes:")
    print(f"Training: {len(train_dataset)} samples")
    print(f"Validation: {len(val_dataset)} samples")

    model = AgeClassifierCNN(num_classes=7, dropout_rate=0.4, use_pretrained=True)
    criterion = CombinedLoss(alpha=0.8, beta=0.2)
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.feature_processor.parameters()) + \
                  list(model.regression_head.parameters()) + \
                  list(model.classification_head.parameters())
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': LEARNING_RATE * 0.1},
        {'params': head_params, 'lr': LEARNING_RATE}
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                   patience=8, min_lr=1e-7)
    print("\nStarting training...")
    history = train_model(model, train_loader, val_loader, criterion, optimizer,
                         scheduler, device, NUM_EPOCHS, PATIENCE)
    print("\nTraining completed!")
    plot_training_history(history)
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    if USE_TTA:
        print("\nEvaluating with Test Time Augmentation...")
        predictions, targets, mae, rmse = evaluate_model_with_tta(model, val_loader, device, n_tta=8)
    else:
        print("\nEvaluating without TTA...")
        predictions, targets, mae, rmse = evaluate_model(model, val_loader, device)
    plot_predictions(predictions, targets)
    print(f"\nDetailed Analysis:")
    print(f"Best Training MAE: {min(history['train_maes']):.2f} years")
    print(f"Best Validation MAE: {min(history['val_maes']):.2f} years")
    print(f"Final Validation MAE: {mae:.2f} years")
    print(f"\nPer-Age-Group Analysis:")
    for age in [20, 30, 40, 50, 60, 70, 80]:
        mask = np.abs(targets - age) < 5
        if np.sum(mask) > 0:
            age_mae = np.mean(np.abs(targets[mask] - predictions[mask]))
            print(f"  Age {age}: MAE = {age_mae:.2f} years ({np.sum(mask)} samples)")
    torch.save(model.state_dict(), 'models/final_age_model.pth')
    print(f"\nModel saved as 'models/final_age_model.pth'")

if __name__ == "__main__":
    main()