# Training loop and logic 
import torch
import time
from sklearn.metrics import mean_absolute_error

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=50, patience=10):
    """Training loop with early stopping"""
    model.to(device)
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    best_val_loss = float('inf')
    patience_counter = 0
    print(f"Training on {device}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        train_reg_loss = 0.0
        train_cls_loss = 0.0
        train_predictions = []
        train_targets = []
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            cls_targets = torch.floor((targets - 20) / 10).long().clamp(0, 6)
            optimizer.zero_grad()
            reg_pred, cls_pred = model(data, mode='both')
            loss, reg_loss, cls_loss = criterion(reg_pred, cls_pred, targets, cls_targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_reg_loss += reg_loss.item()
            train_cls_loss += cls_loss.item()
            train_predictions.extend(reg_pred.detach().cpu().numpy())
            train_targets.extend(targets.detach().cpu().numpy())
        model.eval()
        val_loss = 0.0
        val_reg_loss = 0.0
        val_cls_loss = 0.0
        val_predictions = []
        val_targets_list = []
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                cls_targets = torch.floor((targets - 20) / 10).long().clamp(0, 6)
                reg_pred, cls_pred = model(data, mode='both')
                loss, reg_loss, cls_loss = criterion(reg_pred, cls_pred, targets, cls_targets)
                val_loss += loss.item()
                val_reg_loss += reg_loss.item()
                val_cls_loss += cls_loss.item()
                val_predictions.extend(reg_pred.cpu().numpy())
                val_targets_list.extend(targets.cpu().numpy())
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_reg_loss /= len(train_loader)
        val_reg_loss /= len(val_loader)
        train_cls_loss /= len(train_loader)
        val_cls_loss /= len(val_loader)
        train_mae = mean_absolute_error(train_targets, train_predictions)
        val_mae = mean_absolute_error(val_targets_list, val_predictions)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_maes.append(train_mae)
        val_maes.append(val_mae)
        scheduler.step(val_loss)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s)")
        print(f"  Train Loss: {train_loss:.4f} (Reg: {train_reg_loss:.4f}, Cls: {train_cls_loss:.4f})")
        print(f"  Val Loss: {val_loss:.4f} (Reg: {val_reg_loss:.4f}, Cls: {val_cls_loss:.4f})")
        print(f"  Train MAE: {train_mae:.2f}, Val MAE: {val_mae:.2f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_age_model.pth')
            print(f"  *** New best model saved! ***")
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
        print("-" * 50)
    model.load_state_dict(torch.load('best_age_model.pth'))
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_maes': train_maes,
        'val_maes': val_maes
    } 