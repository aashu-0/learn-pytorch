
import torchmetrics
import torch
from tqdm.auto import tqdm
from typing import List, Dict, Tuple


# train step func
def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    accuracy_fn: torchmetrics.Accuracy,
    device: torch.device)-> Tuple[float, float]:

  # to train mode
  model.train()

  train_loss, train_acc = 0,0

  # loop through each batch
  for batch, (X,y) in enumerate(dataloader):
    #.to(device)
    X, y = X.to(device), y.to(device)

    # do the forward pass
    y_pred = model(X)

    # calculate the loss
    loss = loss_fn(y_pred, y)
    train_loss += loss.item()

    # optimizer zero grad
    optimizer.zero_grad()

    # loss backward (backprop)
    loss.backward()

    # optimizer step (grad descent)
    optimizer.step()

    # accuracy
    train_acc += accuracy_fn(y_pred, y)


  # avg per batch
  train_loss /=len(dataloader)
  train_acc /= len(dataloader)
  return train_loss, train_acc


# test step func
def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    accuracy_fn: torchmetrics.Accuracy,
    device: torch.device) -> Tuple[float, float]:

  # to eval mode
  model.eval()

  test_loss, test_acc = 0,0

  with torch.inference_mode():
    for batch, (X, y) in enumerate(dataloader):

      # to.device
      X,y = X.to(device), y.to(device)

      # do forward pass -> raw logits
      test_pred_logits = model(X)

      # calculate the loss
      loss = loss_fn(test_pred_logits, y)
      test_loss += loss.item()

      # accuracy
      test_acc += accuracy_fn(test_pred_logits, y)

  # avg
  test_loss = test_loss/ len(dataloader)
  test_acc = test_acc/ len(dataloader)
  return test_loss, test_acc


#train func
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          accuracy_fn: torchmetrics.Accuracy,
          epochs: int,
          device= torch.device)-> dict[str, List]:

  # empty result dict
  results = {'train_loss': [],
             'train_acc': [],
             'test_loss': [],
             'test_acc': []}
  # loop
  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_step(model=model,
                                       dataloader = train_dataloader,
                                       loss_fn = loss_fn,
                                       optimizer= optimizer,
                                       accuracy_fn=accuracy_fn,
                                       device = device)

    test_loss, test_acc = test_step(model = model,
                                    dataloader = test_dataloader,
                                    loss_fn = loss_fn,
                                    accuracy_fn=accuracy_fn,
                                    device = device)
    # print out what's happening
    print(f'Epoch: {epoch} | Train Loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test acc: {test_acc:.4f}')

    # update the dict
    results['train_loss'].append(train_loss)
    results['train_acc'].append(train_acc)
    results['test_loss'].append(test_loss)
    results['test_acc'].append(test_acc)

  #return the end results
  return results
