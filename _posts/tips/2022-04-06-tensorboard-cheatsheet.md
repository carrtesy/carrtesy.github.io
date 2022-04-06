---
title: "Tensorboard cheatsheet"
date: 2022-04-06
categories:
 - programming tools and environments

tags:
 - tensorboard
 - torch
---

Here are some useful tips to use tensorboard. 


## Logging into tensorboard

```python
writer = SummaryWriter(log_dir="exp_base")

writer.add_scalar("learning rate", model_optim.param_groups[0]["lr"]) # one variable
writer.add_scalars( # multiple variables
    "Loss Plot",
    {
        "train_loss": train_loss,
        "val_loss": vali_loss,
        "test_loss": test_loss,
    })
```

## check in web browser
```
tensorboard --bind_all --logdir=runs
```