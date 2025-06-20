# Evaluation logic for the model 
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(model, test_loader, device):
    """Evaluate model performance"""
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data, mode='regression')
            predictions.extend(output.cpu().numpy())
            targets.extend(target.numpy())
    predictions = np.array(predictions)
    targets = np.array(targets)
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    print(f"Test Results:")
    print(f"  MAE: {mae:.2f} years")
    print(f"  RMSE: {rmse:.2f} years")
    pred_groups = np.round((predictions - 20) / 10).astype(int)
    true_groups = np.round((targets - 20) / 10).astype(int)
    group_accuracy = np.mean(pred_groups == true_groups)
    print(f"  Age Group Accuracy: {group_accuracy:.3f}")
    return predictions, targets, mae, rmse

def evaluate_model_with_tta(model, val_loader, device, n_tta=8):
    """Evaluate model with Test Time Augmentation (TTA)."""
    model.eval()
    images = []
    targets = []
    for data, target in val_loader:
        images.append(data)
        targets.append(target)
    images = torch.cat(images, dim=0)
    targets = torch.cat(targets, dim=0)
    images_tta = images.repeat((n_tta, 1, 1, 1))
    targets_tta = targets.repeat(n_tta)
    with torch.no_grad():
        predictions = []
        for i in range(0, len(images_tta), val_loader.batch_size):
            batch = images_tta[i:i+val_loader.batch_size].to(device)
            output = model(batch, mode='regression')
            predictions.append(output.cpu())
        predictions = torch.cat(predictions, dim=0)
    num_samples = len(targets)
    predictions = predictions.view(n_tta, num_samples)
    predictions_mean = predictions.mean(dim=0).numpy()
    targets_np = targets.numpy()
    mae = mean_absolute_error(targets_np, predictions_mean)
    mse = mean_squared_error(targets_np, predictions_mean)
    rmse = np.sqrt(mse)
    print(f"Test Results (TTA):")
    print(f"  MAE: {mae:.2f} years")
    print(f"  RMSE: {rmse:.2f} years")
    pred_groups = np.round((predictions_mean - 20) / 10).astype(int)
    true_groups = np.round((targets_np - 20) / 10).astype(int)
    group_accuracy = np.mean(pred_groups == true_groups)
    print(f"  Age Group Accuracy: {group_accuracy:.3f}")
    return predictions_mean, targets_np, mae, rmse 