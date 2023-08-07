import torch
from torch.nn import functional
import copy


class EarlyStopping():
  def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
    self.patience = patience
    self.min_delta = min_delta
    self.restore_best_weights = restore_best_weights
    self.best_model = None
    self.best_loss = None
    self.counter = 0
    self.status = ""
    
  def __call__(self, model, val_loss):
    if self.best_loss == None:
      self.best_loss = val_loss
      self.best_model = copy.deepcopy(model)

    elif self.best_loss - val_loss > self.min_delta:
      self.best_loss = val_loss
      self.counter = 0
      self.best_model.load_state_dict(model.state_dict())

    elif self.best_loss - val_loss <= self.min_delta:
      self.counter += 1

      if self.counter >= self.patience:
        self.status = f"Stopped on {self.counter}"

        if self.restore_best_weights:
          model.load_state_dict(self.best_model.state_dict())

        return True
      
    self.status = f"{self.counter}/{self.patience}"

    return False


#####################################################################


def train_UNet(model, device, optimizer, train_dataloader, val_dataloader=None, early_stopping=None, scheduler=None, epochs=20):
    """
    Training function for UNet Neural Network.

    Inputs:
        - model: UNet.
        - device: cuda if available or cpu.
        - optimizer:
        - train_dataloader: should be created from Cell_Challenge_Segmentation_Dataset().
        - val_dataloader: optional. If present, create it same way as train_dataloader.
        - early_stopping: optional in case one wants to stop training when the validation
                          error stops decending after some epochs.
        - scheduler:
        - epochs:
    
    Outputs:
        - train_losses: list containing training loss through epochs (not baches)
        - val_losses: list containing validation loss through epochs (not baches)
    
    Remarks: during training, training and validation accuracies are also displayed in the
             screen as well as Dice and IoU coefficients calculated with the validation set.
    """

    train_losses = []
    val_losses = []
    
    model = model.to(device=device)

    for epoch in range(1, epochs+1):
        train_epoch_loss = 0
        train_epoch_correct = 0
        train_epoch_total = 0

        model.train()
        for train_idx, (x,y) in enumerate(train_dataloader, start=1):
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long).squeeze(1)
            pred = model(x)

            # Compute loss and gradient descent
            loss = functional.cross_entropy(input=pred, target=y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if scheduler: 
                scheduler.step()

            # Calculate batch loss
            batch_loss = loss.item() # loss.cpu().detach().numpy()
            train_epoch_loss += batch_loss
            
            train_predictions = torch.argmax(pred, dim=1)
            train_epoch_correct += (train_predictions == y).sum()
            train_epoch_total += torch.numel(train_predictions)
        
        # Epoch loss and accuracy
        train_epoch_loss /= len(train_dataloader)
        train_losses.append(train_epoch_loss)
        train_epoch_accuracy = train_epoch_correct / train_epoch_total

        if val_dataloader is not None:
            val_epoch_loss = 0
            val_epoch_correct = 0
            val_epoch_total = 0

            intersection = 0
            denom = 0
            union = 0

            model.eval()
            with torch.no_grad():
                for val_idx, (x, y) in enumerate(val_dataloader, start=1):
                    x = x.to(device=device, dtype=torch.float32)
                    y = y.to(device=device, dtype=torch.long).squeeze(1)
                    pred = model(x)

                    loss = functional.cross_entropy(input=pred, target=y)

                    batch_loss = loss.item() # loss.cpu().detach().numpy()
                    val_epoch_loss += batch_loss
                        
                    val_predictions = torch.argmax(pred, dim=1)
                    val_epoch_correct += (val_predictions == y).sum()
                    val_epoch_total += torch.numel(val_predictions)
                
                    # Dice coefficient
                    intersection += (val_predictions * y).sum()
                    denom += (val_predictions + y).sum()
                    dice = 2*intersection/(denom + 1e-8)

                    # Intersection over Union
                    union += ((val_predictions) + y - (val_predictions * y)).sum()
                    iou = (intersection)/(union + 1e-8)
                
                val_epoch_loss /= len(val_dataloader)
                val_losses.append(val_epoch_loss)
                val_epoch_accuracy = val_epoch_correct / val_epoch_total

                if early_stopping(model, val_epoch_loss):
                   return train_losses, val_losses
            
            print(f'Epoch: {epoch}/{epochs}, Train loss: {train_epoch_loss:.4f}, Val loss: {val_epoch_loss:.4f}, '
                    f'Train acc: {train_epoch_accuracy:.4f}, Val acc: {val_epoch_accuracy:.4f}, '
                    f'Dice: {dice:.4}, IoU: {iou:.4}')
        
        else:
            print(f'Epoch: {epoch}/{epochs}, Train loss: {train_epoch_loss:.4f}, Train acc: {train_epoch_accuracy:.4f}')    

    return train_losses, val_losses


#####################################################################


def train_MaskRCNN(model, device, optimizer, train_dataloader, val_dataloader=None, early_stopping=None, scheduler=None, epochs=20):
    """
    Training function for Mask RCNN Neural Network pretrained model from PyTorch.

    Inputs:
        - model: 
        - device: cuda if available or cpu.
        - optimizer:
        - train_dataloader: should be created from Cell_Challenge_MaskRCNN_Dataset(Dataset).
        - val_dataloader: optional. If present, create it same way as train_dataloader.
        - early_stopping: optional in case one wants to stop training when the validation
                          error stops decending after some epochs.
        - scheduler:
        - epochs:
    
    Outputs:
        - train_losses: list containing training loss through epochs (not baches)
        - val_losses: list containing validation loss through epochs (not baches)
    
    Remarks: only validation accuracies are calculated as well as Dice and IoU coefficients
             using the validation set.
    """

    train_losses = []
    val_losses = []

    for epoch in range(1, epochs+1):
        train_epoch_loss = 0

        model.train()
        for train_idx, (imgs, targs) in enumerate(train_dataloader, start=1):
            imgs = [img.to(device) for img in imgs]
            targs = [{k: v.to(device) for k, v in t.items()} for t in targs]

            losses = model(imgs , targs)
            loss = sum([l for l in losses.values()])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler: 
                scheduler.step()

            batch_loss = loss.item() # loss.cpu().detach().numpy()
            train_epoch_loss += batch_loss

            # print(f"Train Batch [{train_idx}/{len(train_dataloader)}] Loss: {batch_loss:.4f}")
        
        train_epoch_loss /= len(train_dataloader)
        train_losses.append(train_epoch_loss)

        if val_dataloader is not None:
            val_epoch_loss = 0
            val_epoch_correct = 0
            val_epoch_total = 0

            intersection = 0
            denom = 0
            union = 0

            with torch.no_grad():
                for val_idx, (imgs, targs) in enumerate(val_dataloader, start=1):
                    imgs = [img.to(device) for img in imgs]
                    targs = [{k: v.to(device) for k, v in t.items()} for t in targs]

                    model.train()
                    losses = model(imgs, targs)
                    loss = sum([l for l in losses.values()])

                    batch_loss = loss.item() # loss.cpu().detach().numpy()
                    val_epoch_loss += batch_loss

                    # print(f"Val Batch [{val_idx}/{len(val_dataloader)}] Loss: {batch_loss:.4f}")

                    model.eval()
                    pred = model(imgs)

                    gt_mask = [torch.sum(item['masks'], dim=0) for item in targs]
                    total_gt_mask = sum(gt_mask) > 0 # shape (443, 512)

                    pred_mask = [torch.sum(item['masks'], dim=(0,1)) for item in pred]
                    total_pred_mask = sum(pred_mask) > 0 # shape (443, 512)

                    total_gt_mask = total_gt_mask.to(torch.int)
                    total_pred_mask = total_pred_mask.to(torch.int)

                    batch_correct = torch.sum(total_gt_mask == total_pred_mask)
                    val_epoch_correct += batch_correct

                    batch_total = torch.numel(total_gt_mask)
                    val_epoch_total += batch_total

                    # dice coefficient
                    intersection += torch.sum(total_gt_mask * total_pred_mask)
                    denom += torch.sum(total_gt_mask + total_pred_mask)
                    dice = 2 * intersection / (denom + 1e-8)

                    # intersection over union
                    union += torch.sum((total_gt_mask + total_pred_mask) - (total_gt_mask * total_pred_mask))
                    iou = intersection / (union + 1e-8)
                
                val_epoch_loss /= len(val_dataloader)
                val_losses.append(val_epoch_loss)
                val_epoch_accuracy = val_epoch_correct / val_epoch_total

                if early_stopping(model, val_epoch_loss):
                   return train_losses, val_losses

            print(f'Epoch: {epoch}/{epochs}, Train loss: {train_epoch_loss:.4f}, Val loss: {val_epoch_loss:.4f}, '
                    f'Val acc: {val_epoch_accuracy:.4f}, Dice: {dice:.4}, IoU: {iou:.4}')
            
        else:
            print(f'Epoch: {epoch}/{epochs}, Train loss: {train_epoch_loss:.4f}')    

    return train_losses, val_losses