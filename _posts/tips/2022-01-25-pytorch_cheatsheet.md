---
title: "pytorch_cheatsheet"
date: 2022-01-25
last_modified_at: 2022-01-25
categories:
 - pytorch
tags:
 - pytorch
 - tips
 
use_math: true
---
This post is about pytorch cheatsheet.
Commonly used codes and flows are introduced to boost **my** performance!

## Dataset, DataLoaders


```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.Tensor(X.values.reshape(-1, 1, 28, 28))
        self.y = torch.LongTensor(y.values)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
datasets["train"] = MyDataset(X_train, y_train)
dataloaders["train"] = DataLoader(datasets["train"], batch_size=BATCH_SIZE, shuffle=True)
```

## Defining Model


```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        # conv layers (conv - bn - relu - pool)
        self.conv2d = nn.Conv2d(in_channels = 1, out_channels = 2, kernel_size = 3, stride = 1, padding = 1)
        self.bn = nn.BatchNorm2d(num_features = 2)
        self.relu = nn.ReLU()
        self.pool2d = nn.MaxPool2d(kernel_size = 2)

        # linear
        self.linear = nn.Linear(in_features, out_features)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = x
        return out
```

## Model to device


```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNISTModel()
model.to(device)
```

## Train and save your best model


```python
PATH = "best_model.pth"
torch.save(model.state_dict(), PATH)
```



## Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
```

## Load your best model


```python
best_model = Model()
best_model.load_state_dict(torch.load(PATH))
best_model.to(device)
```

## Tensorboard


```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_image('images', grid, 0)
    writer.add_graph(model, images)
    writer.close()
```
