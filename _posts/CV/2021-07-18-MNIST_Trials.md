---
​---
title: "Computer Vision: Try MNIST with Keras"
date: 2021-07-18
last_modified_at: 2021-07-18
categories:
 - computer vision

tags:
 - mnist
 - classification
 - keras
​---
---

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
```


```python
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
```

```console
Found GPU at: /device:GPU:0

```


```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
```


```python
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state = 42)
```


```python
x_train.shape
```




```console
(54000, 28, 28)
```




```python
y_train.shape
```




```console
(54000,)
```




```python
x_val.shape
```




```console
(6000, 28, 28)
```




```python
y_val.shape
```




```console
(6000,)
```




```python
x_test.shape
```




```console
(10000, 28, 28)
```




```python
y_test.shape
```




```console
(10000,)
```




```python
x_train[0]
```




```console
array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         52, 250,  48,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         83, 254,  84,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         83, 254,  48,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         83, 255,  84,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        128, 254,  48,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        178, 254,  48,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        178, 254, 140,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        178, 254, 143,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        112, 254, 143,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        155, 254, 143,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         83, 254, 143,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         83, 254, 115,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         38, 247, 101,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         66, 252, 143,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         59, 251, 143,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0, 226, 195,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0, 144, 251,  64,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0, 144, 254,  82,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0, 144, 254,  82,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,  54, 194,  51,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0]], dtype=uint8)
```




```python
y_train[0]
```




```console
1
```




```python
x_train, x_val, x_test = x_train / 255.0, x_val / 255.0, x_test / 255.0
```


```python
y_train, y_val, y_test = tf.keras.utils.to_categorical(y_train), tf.keras.utils.to_categorical(y_val), tf.keras.utils.to_categorical(y_test)
```


```python
y_train[0]
```




```console
array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)
```




```python
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(x_train[i], cmap = "gray")
plt.show()
```


![png](../../assets/images/cv/MNIST_Trials_files/MNIST_Trials_16_0.png)


## Models


```python
BATCH_SIZE = 64
EPOCHS = 100
```


```python
loss = keras.losses.CategoricalCrossentropy()
```


```python
metrics = ["accuracy"]
```


```python
optimizer = keras.optimizers.SGD(learning_rate = 1e-03)
```

### Linear Model


```python
linear_model = keras.models.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(10, activation="softmax")
])
```


```python
linear_model.summary()
```

```console
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten (Flatten)            (None, 784)               0         
_________________________________________________________________
dense (Dense)                (None, 10)                7850      
=================================================================
Total params: 7,850
Trainable params: 7,850
Non-trainable params: 0
_________________________________________________________________

```


```python
linear_model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
```


```python
with tf.device(device_name = device_name):
    hist = linear_model.fit(x_train, y_train, batch_size = BATCH_SIZE, epochs = EPOCHS, shuffle = True, verbose = 2, validation_data=(x_val, y_val))
```

```console
Epoch 1/100
844/844 - 2s - loss: 1.9300 - accuracy: 0.4194 - val_loss: 1.6387 - val_accuracy: 0.6260
Epoch 2/100
844/844 - 1s - loss: 1.4427 - accuracy: 0.7056 - val_loss: 1.2882 - val_accuracy: 0.7503
Epoch 3/100
844/844 - 1s - loss: 1.1725 - accuracy: 0.7759 - val_loss: 1.0842 - val_accuracy: 0.7900
Epoch 4/100
844/844 - 1s - loss: 1.0094 - accuracy: 0.8051 - val_loss: 0.9542 - val_accuracy: 0.8117
Epoch 5/100
844/844 - 1s - loss: 0.9020 - accuracy: 0.8217 - val_loss: 0.8649 - val_accuracy: 0.8233
Epoch 6/100
844/844 - 1s - loss: 0.8261 - accuracy: 0.8317 - val_loss: 0.7998 - val_accuracy: 0.8313
Epoch 7/100
844/844 - 1s - loss: 0.7696 - accuracy: 0.8386 - val_loss: 0.7500 - val_accuracy: 0.8385
Epoch 8/100
844/844 - 1s - loss: 0.7258 - accuracy: 0.8440 - val_loss: 0.7107 - val_accuracy: 0.8447
Epoch 9/100
844/844 - 1s - loss: 0.6906 - accuracy: 0.8483 - val_loss: 0.6787 - val_accuracy: 0.8502
Epoch 10/100
844/844 - 1s - loss: 0.6618 - accuracy: 0.8523 - val_loss: 0.6520 - val_accuracy: 0.8527
Epoch 11/100
844/844 - 1s - loss: 0.6376 - accuracy: 0.8556 - val_loss: 0.6295 - val_accuracy: 0.8558
Epoch 12/100
844/844 - 1s - loss: 0.6170 - accuracy: 0.8587 - val_loss: 0.6102 - val_accuracy: 0.8592
Epoch 13/100
844/844 - 1s - loss: 0.5991 - accuracy: 0.8608 - val_loss: 0.5933 - val_accuracy: 0.8615
Epoch 14/100
844/844 - 1s - loss: 0.5836 - accuracy: 0.8628 - val_loss: 0.5784 - val_accuracy: 0.8642
Epoch 15/100
844/844 - 1s - loss: 0.5698 - accuracy: 0.8647 - val_loss: 0.5652 - val_accuracy: 0.8672
Epoch 16/100
844/844 - 1s - loss: 0.5575 - accuracy: 0.8663 - val_loss: 0.5534 - val_accuracy: 0.8700
Epoch 17/100
844/844 - 1s - loss: 0.5465 - accuracy: 0.8679 - val_loss: 0.5427 - val_accuracy: 0.8713
Epoch 18/100
844/844 - 1s - loss: 0.5366 - accuracy: 0.8689 - val_loss: 0.5331 - val_accuracy: 0.8720
Epoch 19/100
844/844 - 1s - loss: 0.5275 - accuracy: 0.8702 - val_loss: 0.5243 - val_accuracy: 0.8728
Epoch 20/100
844/844 - 1s - loss: 0.5192 - accuracy: 0.8716 - val_loss: 0.5162 - val_accuracy: 0.8735
Epoch 21/100
844/844 - 1s - loss: 0.5116 - accuracy: 0.8727 - val_loss: 0.5086 - val_accuracy: 0.8742
Epoch 22/100
844/844 - 1s - loss: 0.5046 - accuracy: 0.8741 - val_loss: 0.5018 - val_accuracy: 0.8763
Epoch 23/100
844/844 - 1s - loss: 0.4981 - accuracy: 0.8745 - val_loss: 0.4954 - val_accuracy: 0.8772
Epoch 24/100
844/844 - 1s - loss: 0.4920 - accuracy: 0.8757 - val_loss: 0.4894 - val_accuracy: 0.8772
Epoch 25/100
844/844 - 1s - loss: 0.4864 - accuracy: 0.8766 - val_loss: 0.4839 - val_accuracy: 0.8780
Epoch 26/100
844/844 - 1s - loss: 0.4811 - accuracy: 0.8771 - val_loss: 0.4787 - val_accuracy: 0.8798
Epoch 27/100
844/844 - 1s - loss: 0.4762 - accuracy: 0.8778 - val_loss: 0.4737 - val_accuracy: 0.8803
Epoch 28/100
844/844 - 1s - loss: 0.4715 - accuracy: 0.8787 - val_loss: 0.4691 - val_accuracy: 0.8815
Epoch 29/100
844/844 - 1s - loss: 0.4671 - accuracy: 0.8791 - val_loss: 0.4648 - val_accuracy: 0.8818
Epoch 30/100
844/844 - 1s - loss: 0.4630 - accuracy: 0.8800 - val_loss: 0.4607 - val_accuracy: 0.8825
Epoch 31/100
844/844 - 1s - loss: 0.4590 - accuracy: 0.8805 - val_loss: 0.4567 - val_accuracy: 0.8845
Epoch 32/100
844/844 - 1s - loss: 0.4553 - accuracy: 0.8813 - val_loss: 0.4530 - val_accuracy: 0.8843
Epoch 33/100
844/844 - 1s - loss: 0.4518 - accuracy: 0.8820 - val_loss: 0.4494 - val_accuracy: 0.8842
Epoch 34/100
844/844 - 1s - loss: 0.4484 - accuracy: 0.8829 - val_loss: 0.4461 - val_accuracy: 0.8853
Epoch 35/100
844/844 - 1s - loss: 0.4452 - accuracy: 0.8837 - val_loss: 0.4429 - val_accuracy: 0.8862
Epoch 36/100
844/844 - 1s - loss: 0.4421 - accuracy: 0.8840 - val_loss: 0.4398 - val_accuracy: 0.8863
Epoch 37/100
844/844 - 1s - loss: 0.4392 - accuracy: 0.8846 - val_loss: 0.4368 - val_accuracy: 0.8867
Epoch 38/100
844/844 - 1s - loss: 0.4364 - accuracy: 0.8852 - val_loss: 0.4340 - val_accuracy: 0.8875
Epoch 39/100
844/844 - 1s - loss: 0.4337 - accuracy: 0.8856 - val_loss: 0.4313 - val_accuracy: 0.8875
Epoch 40/100
844/844 - 1s - loss: 0.4311 - accuracy: 0.8861 - val_loss: 0.4287 - val_accuracy: 0.8880
Epoch 41/100
844/844 - 1s - loss: 0.4286 - accuracy: 0.8862 - val_loss: 0.4262 - val_accuracy: 0.8882
Epoch 42/100
844/844 - 1s - loss: 0.4262 - accuracy: 0.8866 - val_loss: 0.4239 - val_accuracy: 0.8887
Epoch 43/100
844/844 - 1s - loss: 0.4239 - accuracy: 0.8872 - val_loss: 0.4215 - val_accuracy: 0.8888
Epoch 44/100
844/844 - 1s - loss: 0.4217 - accuracy: 0.8875 - val_loss: 0.4194 - val_accuracy: 0.8895
Epoch 45/100
844/844 - 1s - loss: 0.4196 - accuracy: 0.8880 - val_loss: 0.4172 - val_accuracy: 0.8897
Epoch 46/100
844/844 - 1s - loss: 0.4175 - accuracy: 0.8884 - val_loss: 0.4151 - val_accuracy: 0.8903
Epoch 47/100
844/844 - 2s - loss: 0.4155 - accuracy: 0.8890 - val_loss: 0.4131 - val_accuracy: 0.8905
Epoch 48/100
844/844 - 1s - loss: 0.4136 - accuracy: 0.8891 - val_loss: 0.4111 - val_accuracy: 0.8908
Epoch 49/100
844/844 - 1s - loss: 0.4117 - accuracy: 0.8894 - val_loss: 0.4093 - val_accuracy: 0.8903
Epoch 50/100
844/844 - 1s - loss: 0.4099 - accuracy: 0.8899 - val_loss: 0.4075 - val_accuracy: 0.8905
Epoch 51/100
844/844 - 1s - loss: 0.4082 - accuracy: 0.8905 - val_loss: 0.4057 - val_accuracy: 0.8913
Epoch 52/100
844/844 - 1s - loss: 0.4065 - accuracy: 0.8910 - val_loss: 0.4040 - val_accuracy: 0.8920
Epoch 53/100
844/844 - 1s - loss: 0.4048 - accuracy: 0.8912 - val_loss: 0.4023 - val_accuracy: 0.8922
Epoch 54/100
844/844 - 1s - loss: 0.4032 - accuracy: 0.8916 - val_loss: 0.4007 - val_accuracy: 0.8930
Epoch 55/100
844/844 - 1s - loss: 0.4017 - accuracy: 0.8916 - val_loss: 0.3992 - val_accuracy: 0.8932
Epoch 56/100
844/844 - 1s - loss: 0.4002 - accuracy: 0.8921 - val_loss: 0.3976 - val_accuracy: 0.8933
Epoch 57/100
844/844 - 1s - loss: 0.3987 - accuracy: 0.8925 - val_loss: 0.3962 - val_accuracy: 0.8937
Epoch 58/100
844/844 - 1s - loss: 0.3973 - accuracy: 0.8927 - val_loss: 0.3947 - val_accuracy: 0.8938
Epoch 59/100
844/844 - 1s - loss: 0.3959 - accuracy: 0.8929 - val_loss: 0.3934 - val_accuracy: 0.8942
Epoch 60/100
844/844 - 1s - loss: 0.3945 - accuracy: 0.8934 - val_loss: 0.3920 - val_accuracy: 0.8945
Epoch 61/100
844/844 - 1s - loss: 0.3932 - accuracy: 0.8935 - val_loss: 0.3906 - val_accuracy: 0.8950
Epoch 62/100
844/844 - 1s - loss: 0.3919 - accuracy: 0.8938 - val_loss: 0.3894 - val_accuracy: 0.8952
Epoch 63/100
844/844 - 1s - loss: 0.3907 - accuracy: 0.8941 - val_loss: 0.3881 - val_accuracy: 0.8953
Epoch 64/100
844/844 - 1s - loss: 0.3895 - accuracy: 0.8941 - val_loss: 0.3868 - val_accuracy: 0.8957
Epoch 65/100
844/844 - 1s - loss: 0.3883 - accuracy: 0.8945 - val_loss: 0.3857 - val_accuracy: 0.8958
Epoch 66/100
844/844 - 1s - loss: 0.3871 - accuracy: 0.8947 - val_loss: 0.3845 - val_accuracy: 0.8967
Epoch 67/100
844/844 - 1s - loss: 0.3859 - accuracy: 0.8950 - val_loss: 0.3834 - val_accuracy: 0.8973
Epoch 68/100
844/844 - 1s - loss: 0.3848 - accuracy: 0.8956 - val_loss: 0.3822 - val_accuracy: 0.8967
Epoch 69/100
844/844 - 1s - loss: 0.3837 - accuracy: 0.8957 - val_loss: 0.3811 - val_accuracy: 0.8973
Epoch 70/100
844/844 - 1s - loss: 0.3827 - accuracy: 0.8959 - val_loss: 0.3801 - val_accuracy: 0.8973
Epoch 71/100
844/844 - 1s - loss: 0.3816 - accuracy: 0.8961 - val_loss: 0.3790 - val_accuracy: 0.8975
Epoch 72/100
844/844 - 1s - loss: 0.3806 - accuracy: 0.8964 - val_loss: 0.3780 - val_accuracy: 0.8975
Epoch 73/100
844/844 - 1s - loss: 0.3796 - accuracy: 0.8966 - val_loss: 0.3770 - val_accuracy: 0.8977
Epoch 74/100
844/844 - 1s - loss: 0.3786 - accuracy: 0.8969 - val_loss: 0.3760 - val_accuracy: 0.8980
Epoch 75/100
844/844 - 1s - loss: 0.3777 - accuracy: 0.8968 - val_loss: 0.3751 - val_accuracy: 0.8987
Epoch 76/100
844/844 - 1s - loss: 0.3767 - accuracy: 0.8974 - val_loss: 0.3741 - val_accuracy: 0.8987
Epoch 77/100
844/844 - 1s - loss: 0.3758 - accuracy: 0.8975 - val_loss: 0.3732 - val_accuracy: 0.8990
Epoch 78/100
844/844 - 1s - loss: 0.3749 - accuracy: 0.8979 - val_loss: 0.3723 - val_accuracy: 0.8995
Epoch 79/100
844/844 - 1s - loss: 0.3740 - accuracy: 0.8979 - val_loss: 0.3714 - val_accuracy: 0.8998
Epoch 80/100
844/844 - 1s - loss: 0.3732 - accuracy: 0.8982 - val_loss: 0.3705 - val_accuracy: 0.8997
Epoch 81/100
844/844 - 1s - loss: 0.3723 - accuracy: 0.8985 - val_loss: 0.3697 - val_accuracy: 0.8998
Epoch 82/100
844/844 - 1s - loss: 0.3715 - accuracy: 0.8985 - val_loss: 0.3688 - val_accuracy: 0.9003
Epoch 83/100
844/844 - 1s - loss: 0.3707 - accuracy: 0.8985 - val_loss: 0.3680 - val_accuracy: 0.9005
Epoch 84/100
844/844 - 1s - loss: 0.3699 - accuracy: 0.8986 - val_loss: 0.3672 - val_accuracy: 0.9005
Epoch 85/100
844/844 - 1s - loss: 0.3691 - accuracy: 0.8990 - val_loss: 0.3664 - val_accuracy: 0.9008
Epoch 86/100
844/844 - 1s - loss: 0.3683 - accuracy: 0.8991 - val_loss: 0.3656 - val_accuracy: 0.9008
Epoch 87/100
844/844 - 1s - loss: 0.3675 - accuracy: 0.8991 - val_loss: 0.3649 - val_accuracy: 0.9010
Epoch 88/100
844/844 - 1s - loss: 0.3668 - accuracy: 0.8996 - val_loss: 0.3641 - val_accuracy: 0.9012
Epoch 89/100
844/844 - 1s - loss: 0.3660 - accuracy: 0.8996 - val_loss: 0.3633 - val_accuracy: 0.9012
Epoch 90/100
844/844 - 1s - loss: 0.3653 - accuracy: 0.8999 - val_loss: 0.3627 - val_accuracy: 0.9020
Epoch 91/100
844/844 - 1s - loss: 0.3646 - accuracy: 0.9000 - val_loss: 0.3619 - val_accuracy: 0.9022
Epoch 92/100
844/844 - 1s - loss: 0.3639 - accuracy: 0.9002 - val_loss: 0.3612 - val_accuracy: 0.9025
Epoch 93/100
844/844 - 1s - loss: 0.3632 - accuracy: 0.9004 - val_loss: 0.3605 - val_accuracy: 0.9025
Epoch 94/100
844/844 - 1s - loss: 0.3625 - accuracy: 0.9006 - val_loss: 0.3599 - val_accuracy: 0.9027
Epoch 95/100
844/844 - 1s - loss: 0.3619 - accuracy: 0.9008 - val_loss: 0.3592 - val_accuracy: 0.9027
Epoch 96/100
844/844 - 1s - loss: 0.3612 - accuracy: 0.9009 - val_loss: 0.3585 - val_accuracy: 0.9028
Epoch 97/100
844/844 - 1s - loss: 0.3605 - accuracy: 0.9011 - val_loss: 0.3579 - val_accuracy: 0.9030
Epoch 98/100
844/844 - 1s - loss: 0.3599 - accuracy: 0.9011 - val_loss: 0.3573 - val_accuracy: 0.9028
Epoch 99/100
844/844 - 1s - loss: 0.3593 - accuracy: 0.9014 - val_loss: 0.3567 - val_accuracy: 0.9028
Epoch 100/100
844/844 - 1s - loss: 0.3587 - accuracy: 0.9014 - val_loss: 0.3560 - val_accuracy: 0.9028

```


```python
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper left')

plt.show()
```


![png](MNIST_Trials_files/MNIST_Trials_27_0.png)



```python
with tf.device(device_name = device_name):
    linear_model.evaluate(x_test, y_test, batch_size = BATCH_SIZE, verbose = 2)
```

```console
157/157 - 0s - loss: 0.3404 - accuracy: 0.9067

```


```python
test_image = tf.expand_dims(x_test[0], axis = 0)
```


```python
linear_model_output = linear_model(test_image)
linear_model_output
```




```console
<tf.Tensor: shape=(1, 10), dtype=float32, numpy=
array([[2.1254762e-04, 1.3267899e-06, 2.1704029e-04, 2.0511877e-03,
        6.5497406e-05, 9.2676739e-05, 4.4080762e-06, 9.9426877e-01,
        1.7101581e-04, 2.9154746e-03]], dtype=float32)>
```




```python
softmax_output = tf.nn.softmax(linear_model_output)
softmax_output
```




```console
<tf.Tensor: shape=(1, 10), dtype=float32, numpy=
array([[0.08542631, 0.08540826, 0.08542669, 0.08558352, 0.08541374,
        0.08541606, 0.08540852, 0.23083663, 0.08542276, 0.08565752]],
      dtype=float32)>
```




```python
tf.math.argmax(softmax_output, axis = 1) 
```




```console
<tf.Tensor: shape=(1,), dtype=int64, numpy=array([7])>
```




```python
plt.imshow(x_test[0])
```




```console
<matplotlib.image.AxesImage at 0x7f74065a3b90>
```




![png](../../assets/images/cv/MNIST_Trials_files/MNIST_Trials_33_1.png)


### Multi Layer Linear Model


```python
MLL_model = keras.models.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(256),
    keras.layers.Dense(128),
    keras.layers.Dense(10, activation="softmax"),
])
```


```python
MLL_model.summary()
```

```console
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten_2 (Flatten)          (None, 784)               0         
_________________________________________________________________
dense_4 (Dense)              (None, 256)               200960    
_________________________________________________________________
dense_5 (Dense)              (None, 128)               32896     
_________________________________________________________________
dense_6 (Dense)              (None, 10)                1290      
=================================================================
Total params: 235,146
Trainable params: 235,146
Non-trainable params: 0
_________________________________________________________________

```


```python
MLL_model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
```


```python
with tf.device(device_name = device_name):
    hist = MLL_model.fit(x_train, y_train, batch_size = BATCH_SIZE, epochs = EPOCHS, shuffle = True, verbose = 2, validation_data=(x_val, y_val))
```

```console
Epoch 1/100
844/844 - 2s - loss: 1.4309 - accuracy: 0.6255 - val_loss: 0.9681 - val_accuracy: 0.7858
Epoch 2/100
844/844 - 2s - loss: 0.7993 - accuracy: 0.8141 - val_loss: 0.6930 - val_accuracy: 0.8310
Epoch 3/100
844/844 - 2s - loss: 0.6246 - accuracy: 0.8454 - val_loss: 0.5807 - val_accuracy: 0.8490
Epoch 4/100
844/844 - 2s - loss: 0.5424 - accuracy: 0.8605 - val_loss: 0.5190 - val_accuracy: 0.8608
Epoch 5/100
844/844 - 2s - loss: 0.4940 - accuracy: 0.8695 - val_loss: 0.4790 - val_accuracy: 0.8720
Epoch 6/100
844/844 - 2s - loss: 0.4616 - accuracy: 0.8750 - val_loss: 0.4510 - val_accuracy: 0.8763
Epoch 7/100
844/844 - 2s - loss: 0.4382 - accuracy: 0.8801 - val_loss: 0.4302 - val_accuracy: 0.8813
Epoch 8/100
844/844 - 2s - loss: 0.4205 - accuracy: 0.8844 - val_loss: 0.4140 - val_accuracy: 0.8843
Epoch 9/100
844/844 - 2s - loss: 0.4065 - accuracy: 0.8872 - val_loss: 0.4008 - val_accuracy: 0.8872
Epoch 10/100
844/844 - 2s - loss: 0.3952 - accuracy: 0.8895 - val_loss: 0.3901 - val_accuracy: 0.8908
Epoch 11/100
844/844 - 2s - loss: 0.3857 - accuracy: 0.8918 - val_loss: 0.3813 - val_accuracy: 0.8933
Epoch 12/100
844/844 - 2s - loss: 0.3776 - accuracy: 0.8937 - val_loss: 0.3737 - val_accuracy: 0.8955
Epoch 13/100
844/844 - 2s - loss: 0.3707 - accuracy: 0.8955 - val_loss: 0.3667 - val_accuracy: 0.8970
Epoch 14/100
844/844 - 2s - loss: 0.3647 - accuracy: 0.8968 - val_loss: 0.3611 - val_accuracy: 0.8992
Epoch 15/100
844/844 - 2s - loss: 0.3593 - accuracy: 0.8982 - val_loss: 0.3557 - val_accuracy: 0.9005
Epoch 16/100
844/844 - 2s - loss: 0.3545 - accuracy: 0.8989 - val_loss: 0.3512 - val_accuracy: 0.9023
Epoch 17/100
844/844 - 2s - loss: 0.3503 - accuracy: 0.9002 - val_loss: 0.3470 - val_accuracy: 0.9027
Epoch 18/100
844/844 - 2s - loss: 0.3464 - accuracy: 0.9012 - val_loss: 0.3434 - val_accuracy: 0.9043
Epoch 19/100
844/844 - 2s - loss: 0.3428 - accuracy: 0.9022 - val_loss: 0.3402 - val_accuracy: 0.9050
Epoch 20/100
844/844 - 2s - loss: 0.3396 - accuracy: 0.9027 - val_loss: 0.3368 - val_accuracy: 0.9068
Epoch 21/100
844/844 - 2s - loss: 0.3367 - accuracy: 0.9034 - val_loss: 0.3342 - val_accuracy: 0.9083
Epoch 22/100
844/844 - 2s - loss: 0.3340 - accuracy: 0.9045 - val_loss: 0.3316 - val_accuracy: 0.9093
Epoch 23/100
844/844 - 2s - loss: 0.3314 - accuracy: 0.9050 - val_loss: 0.3291 - val_accuracy: 0.9097
Epoch 24/100
844/844 - 2s - loss: 0.3290 - accuracy: 0.9062 - val_loss: 0.3269 - val_accuracy: 0.9103
Epoch 25/100
844/844 - 2s - loss: 0.3268 - accuracy: 0.9070 - val_loss: 0.3248 - val_accuracy: 0.9108
Epoch 26/100
844/844 - 2s - loss: 0.3248 - accuracy: 0.9074 - val_loss: 0.3229 - val_accuracy: 0.9112
Epoch 27/100
844/844 - 2s - loss: 0.3228 - accuracy: 0.9083 - val_loss: 0.3209 - val_accuracy: 0.9115
Epoch 28/100
844/844 - 2s - loss: 0.3209 - accuracy: 0.9087 - val_loss: 0.3190 - val_accuracy: 0.9123
Epoch 29/100
844/844 - 2s - loss: 0.3191 - accuracy: 0.9093 - val_loss: 0.3176 - val_accuracy: 0.9125
Epoch 30/100
844/844 - 2s - loss: 0.3176 - accuracy: 0.9096 - val_loss: 0.3158 - val_accuracy: 0.9137
Epoch 31/100
844/844 - 2s - loss: 0.3160 - accuracy: 0.9102 - val_loss: 0.3142 - val_accuracy: 0.9132
Epoch 32/100
844/844 - 2s - loss: 0.3145 - accuracy: 0.9105 - val_loss: 0.3134 - val_accuracy: 0.9147
Epoch 33/100
844/844 - 2s - loss: 0.3131 - accuracy: 0.9111 - val_loss: 0.3117 - val_accuracy: 0.9145
Epoch 34/100
844/844 - 2s - loss: 0.3117 - accuracy: 0.9117 - val_loss: 0.3104 - val_accuracy: 0.9142
Epoch 35/100
844/844 - 2s - loss: 0.3104 - accuracy: 0.9123 - val_loss: 0.3095 - val_accuracy: 0.9158
Epoch 36/100
844/844 - 2s - loss: 0.3092 - accuracy: 0.9128 - val_loss: 0.3081 - val_accuracy: 0.9155
Epoch 37/100
844/844 - 2s - loss: 0.3079 - accuracy: 0.9130 - val_loss: 0.3076 - val_accuracy: 0.9157
Epoch 38/100
844/844 - 2s - loss: 0.3068 - accuracy: 0.9136 - val_loss: 0.3061 - val_accuracy: 0.9168
Epoch 39/100
844/844 - 2s - loss: 0.3057 - accuracy: 0.9139 - val_loss: 0.3049 - val_accuracy: 0.9168
Epoch 40/100
844/844 - 2s - loss: 0.3047 - accuracy: 0.9140 - val_loss: 0.3045 - val_accuracy: 0.9158
Epoch 41/100
844/844 - 2s - loss: 0.3036 - accuracy: 0.9146 - val_loss: 0.3035 - val_accuracy: 0.9168
Epoch 42/100
844/844 - 2s - loss: 0.3026 - accuracy: 0.9146 - val_loss: 0.3027 - val_accuracy: 0.9175
Epoch 43/100
844/844 - 2s - loss: 0.3018 - accuracy: 0.9149 - val_loss: 0.3016 - val_accuracy: 0.9178
Epoch 44/100
844/844 - 2s - loss: 0.3008 - accuracy: 0.9154 - val_loss: 0.3009 - val_accuracy: 0.9178
Epoch 45/100
844/844 - 2s - loss: 0.2999 - accuracy: 0.9154 - val_loss: 0.3001 - val_accuracy: 0.9175
Epoch 46/100
844/844 - 2s - loss: 0.2991 - accuracy: 0.9157 - val_loss: 0.2991 - val_accuracy: 0.9175
Epoch 47/100
844/844 - 2s - loss: 0.2983 - accuracy: 0.9161 - val_loss: 0.2986 - val_accuracy: 0.9178
Epoch 48/100
844/844 - 2s - loss: 0.2974 - accuracy: 0.9162 - val_loss: 0.2979 - val_accuracy: 0.9178
Epoch 49/100
844/844 - 2s - loss: 0.2966 - accuracy: 0.9163 - val_loss: 0.2973 - val_accuracy: 0.9173
Epoch 50/100
844/844 - 2s - loss: 0.2959 - accuracy: 0.9166 - val_loss: 0.2966 - val_accuracy: 0.9185
Epoch 51/100
844/844 - 2s - loss: 0.2951 - accuracy: 0.9168 - val_loss: 0.2960 - val_accuracy: 0.9180
Epoch 52/100
844/844 - 2s - loss: 0.2944 - accuracy: 0.9170 - val_loss: 0.2951 - val_accuracy: 0.9180
Epoch 53/100
844/844 - 2s - loss: 0.2937 - accuracy: 0.9173 - val_loss: 0.2951 - val_accuracy: 0.9182
Epoch 54/100
844/844 - 2s - loss: 0.2931 - accuracy: 0.9171 - val_loss: 0.2944 - val_accuracy: 0.9182
Epoch 55/100
844/844 - 2s - loss: 0.2924 - accuracy: 0.9177 - val_loss: 0.2935 - val_accuracy: 0.9177
Epoch 56/100
844/844 - 2s - loss: 0.2918 - accuracy: 0.9179 - val_loss: 0.2929 - val_accuracy: 0.9188
Epoch 57/100
844/844 - 2s - loss: 0.2911 - accuracy: 0.9181 - val_loss: 0.2925 - val_accuracy: 0.9185
Epoch 58/100
844/844 - 2s - loss: 0.2906 - accuracy: 0.9182 - val_loss: 0.2920 - val_accuracy: 0.9190
Epoch 59/100
844/844 - 2s - loss: 0.2899 - accuracy: 0.9184 - val_loss: 0.2913 - val_accuracy: 0.9188
Epoch 60/100
844/844 - 2s - loss: 0.2893 - accuracy: 0.9184 - val_loss: 0.2910 - val_accuracy: 0.9190
Epoch 61/100
844/844 - 2s - loss: 0.2888 - accuracy: 0.9188 - val_loss: 0.2904 - val_accuracy: 0.9198
Epoch 62/100
844/844 - 2s - loss: 0.2882 - accuracy: 0.9190 - val_loss: 0.2903 - val_accuracy: 0.9183
Epoch 63/100
844/844 - 2s - loss: 0.2876 - accuracy: 0.9192 - val_loss: 0.2900 - val_accuracy: 0.9197
Epoch 64/100
844/844 - 2s - loss: 0.2872 - accuracy: 0.9195 - val_loss: 0.2895 - val_accuracy: 0.9190
Epoch 65/100
844/844 - 2s - loss: 0.2866 - accuracy: 0.9193 - val_loss: 0.2890 - val_accuracy: 0.9197
Epoch 66/100
844/844 - 2s - loss: 0.2861 - accuracy: 0.9194 - val_loss: 0.2885 - val_accuracy: 0.9190
Epoch 67/100
844/844 - 2s - loss: 0.2856 - accuracy: 0.9195 - val_loss: 0.2887 - val_accuracy: 0.9200
Epoch 68/100
844/844 - 2s - loss: 0.2852 - accuracy: 0.9199 - val_loss: 0.2877 - val_accuracy: 0.9195
Epoch 69/100
844/844 - 2s - loss: 0.2847 - accuracy: 0.9198 - val_loss: 0.2870 - val_accuracy: 0.9195
Epoch 70/100
844/844 - 2s - loss: 0.2842 - accuracy: 0.9202 - val_loss: 0.2868 - val_accuracy: 0.9208
Epoch 71/100
844/844 - 2s - loss: 0.2838 - accuracy: 0.9203 - val_loss: 0.2866 - val_accuracy: 0.9205
Epoch 72/100
844/844 - 2s - loss: 0.2833 - accuracy: 0.9205 - val_loss: 0.2863 - val_accuracy: 0.9200
Epoch 73/100
844/844 - 2s - loss: 0.2829 - accuracy: 0.9203 - val_loss: 0.2859 - val_accuracy: 0.9205
Epoch 74/100
844/844 - 2s - loss: 0.2825 - accuracy: 0.9204 - val_loss: 0.2855 - val_accuracy: 0.9208
Epoch 75/100
844/844 - 2s - loss: 0.2820 - accuracy: 0.9206 - val_loss: 0.2853 - val_accuracy: 0.9213
Epoch 76/100
844/844 - 2s - loss: 0.2817 - accuracy: 0.9205 - val_loss: 0.2850 - val_accuracy: 0.9215
Epoch 77/100
844/844 - 2s - loss: 0.2813 - accuracy: 0.9210 - val_loss: 0.2845 - val_accuracy: 0.9213
Epoch 78/100
844/844 - 2s - loss: 0.2808 - accuracy: 0.9208 - val_loss: 0.2842 - val_accuracy: 0.9202
Epoch 79/100
844/844 - 2s - loss: 0.2804 - accuracy: 0.9211 - val_loss: 0.2843 - val_accuracy: 0.9213
Epoch 80/100
844/844 - 2s - loss: 0.2800 - accuracy: 0.9216 - val_loss: 0.2835 - val_accuracy: 0.9210
Epoch 81/100
844/844 - 2s - loss: 0.2796 - accuracy: 0.9215 - val_loss: 0.2838 - val_accuracy: 0.9212
Epoch 82/100
844/844 - 2s - loss: 0.2793 - accuracy: 0.9214 - val_loss: 0.2839 - val_accuracy: 0.9202
Epoch 83/100
844/844 - 2s - loss: 0.2790 - accuracy: 0.9218 - val_loss: 0.2830 - val_accuracy: 0.9217
Epoch 84/100
844/844 - 2s - loss: 0.2785 - accuracy: 0.9216 - val_loss: 0.2831 - val_accuracy: 0.9202
Epoch 85/100
844/844 - 2s - loss: 0.2782 - accuracy: 0.9217 - val_loss: 0.2825 - val_accuracy: 0.9205
Epoch 86/100
844/844 - 2s - loss: 0.2778 - accuracy: 0.9221 - val_loss: 0.2822 - val_accuracy: 0.9223
Epoch 87/100
844/844 - 2s - loss: 0.2775 - accuracy: 0.9223 - val_loss: 0.2824 - val_accuracy: 0.9213
Epoch 88/100
844/844 - 2s - loss: 0.2772 - accuracy: 0.9223 - val_loss: 0.2820 - val_accuracy: 0.9212
Epoch 89/100
844/844 - 2s - loss: 0.2768 - accuracy: 0.9225 - val_loss: 0.2818 - val_accuracy: 0.9212
Epoch 90/100
844/844 - 2s - loss: 0.2764 - accuracy: 0.9224 - val_loss: 0.2815 - val_accuracy: 0.9220
Epoch 91/100
844/844 - 2s - loss: 0.2762 - accuracy: 0.9226 - val_loss: 0.2814 - val_accuracy: 0.9223
Epoch 92/100
844/844 - 2s - loss: 0.2759 - accuracy: 0.9228 - val_loss: 0.2810 - val_accuracy: 0.9222
Epoch 93/100
844/844 - 2s - loss: 0.2755 - accuracy: 0.9227 - val_loss: 0.2803 - val_accuracy: 0.9217
Epoch 94/100
844/844 - 2s - loss: 0.2753 - accuracy: 0.9228 - val_loss: 0.2803 - val_accuracy: 0.9218
Epoch 95/100
844/844 - 2s - loss: 0.2749 - accuracy: 0.9231 - val_loss: 0.2801 - val_accuracy: 0.9218
Epoch 96/100
844/844 - 2s - loss: 0.2746 - accuracy: 0.9229 - val_loss: 0.2801 - val_accuracy: 0.9213
Epoch 97/100
844/844 - 2s - loss: 0.2743 - accuracy: 0.9235 - val_loss: 0.2795 - val_accuracy: 0.9217
Epoch 98/100
844/844 - 2s - loss: 0.2740 - accuracy: 0.9230 - val_loss: 0.2796 - val_accuracy: 0.9220
Epoch 99/100
844/844 - 2s - loss: 0.2738 - accuracy: 0.9234 - val_loss: 0.2796 - val_accuracy: 0.9222
Epoch 100/100
844/844 - 2s - loss: 0.2735 - accuracy: 0.9234 - val_loss: 0.2792 - val_accuracy: 0.9225

```


```python
with tf.device(device_name = device_name):
    MLL_model.evaluate(x_test, y_test, batch_size = BATCH_SIZE, verbose = 2)
```

```console
157/157 - 0s - loss: 0.2768 - accuracy: 0.9208

```


```python
def show_model_output(data, model):
    plt.imshow(data)
    expanded_data = tf.expand_dims(data, axis = 0)
    model_output = model(expanded_data)
    return tf.math.argmax(model_output, axis = 1)[0]

show_model_output(x_test[0], MLL_model)
```




```console
<tf.Tensor: shape=(), dtype=int64, numpy=7>
```




![png](MNIST_Trials_files/MNIST_Trials_40_1.png)


### Multi Layer Perceptron (MLP)



```python
MLP_model = keras.models.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(256, activation = "relu"),
    keras.layers.Dense(128, activation = "relu"),
    keras.layers.Dense(10, activation = "softmax"),
])
```


```python
MLP_model.summary()
```

```console
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten_2 (Flatten)          (None, 784)               0         
_________________________________________________________________
dense_4 (Dense)              (None, 256)               200960    
_________________________________________________________________
dense_5 (Dense)              (None, 128)               32896     
_________________________________________________________________
dense_6 (Dense)              (None, 10)                1290      
=================================================================
Total params: 235,146
Trainable params: 235,146
Non-trainable params: 0
_________________________________________________________________

```


```python
MLP_model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
```


```python
with tf.device(device_name = device_name):
    hist = MLP_model.fit(x_train, y_train, batch_size = BATCH_SIZE, epochs = EPOCHS, shuffle = True, verbose = 2, validation_data=(x_val, y_val))
```

```console
Epoch 1/100
844/844 - 2s - loss: 2.0317 - accuracy: 0.4185 - val_loss: 1.7569 - val_accuracy: 0.6358
Epoch 2/100
844/844 - 2s - loss: 1.4830 - accuracy: 0.7230 - val_loss: 1.2276 - val_accuracy: 0.7742
Epoch 3/100
844/844 - 2s - loss: 1.0384 - accuracy: 0.8079 - val_loss: 0.8873 - val_accuracy: 0.8283
Epoch 4/100
844/844 - 2s - loss: 0.7858 - accuracy: 0.8380 - val_loss: 0.7056 - val_accuracy: 0.8488
Epoch 5/100
844/844 - 2s - loss: 0.6485 - accuracy: 0.8545 - val_loss: 0.6018 - val_accuracy: 0.8637
Epoch 6/100
844/844 - 2s - loss: 0.5665 - accuracy: 0.8657 - val_loss: 0.5352 - val_accuracy: 0.8737
Epoch 7/100
844/844 - 2s - loss: 0.5125 - accuracy: 0.8744 - val_loss: 0.4906 - val_accuracy: 0.8775
Epoch 8/100
844/844 - 2s - loss: 0.4743 - accuracy: 0.8809 - val_loss: 0.4563 - val_accuracy: 0.8838
Epoch 9/100
844/844 - 2s - loss: 0.4457 - accuracy: 0.8863 - val_loss: 0.4307 - val_accuracy: 0.8870
Epoch 10/100
844/844 - 2s - loss: 0.4234 - accuracy: 0.8906 - val_loss: 0.4105 - val_accuracy: 0.8918
Epoch 11/100
844/844 - 2s - loss: 0.4055 - accuracy: 0.8940 - val_loss: 0.3936 - val_accuracy: 0.8923
Epoch 12/100
844/844 - 2s - loss: 0.3906 - accuracy: 0.8964 - val_loss: 0.3799 - val_accuracy: 0.8962
Epoch 13/100
844/844 - 2s - loss: 0.3780 - accuracy: 0.8993 - val_loss: 0.3684 - val_accuracy: 0.8982
Epoch 14/100
844/844 - 2s - loss: 0.3671 - accuracy: 0.9014 - val_loss: 0.3575 - val_accuracy: 0.9000
Epoch 15/100
844/844 - 2s - loss: 0.3577 - accuracy: 0.9036 - val_loss: 0.3490 - val_accuracy: 0.9032
Epoch 16/100
844/844 - 2s - loss: 0.3492 - accuracy: 0.9048 - val_loss: 0.3404 - val_accuracy: 0.9050
Epoch 17/100
844/844 - 2s - loss: 0.3415 - accuracy: 0.9066 - val_loss: 0.3332 - val_accuracy: 0.9078
Epoch 18/100
844/844 - 2s - loss: 0.3346 - accuracy: 0.9078 - val_loss: 0.3273 - val_accuracy: 0.9090
Epoch 19/100
844/844 - 2s - loss: 0.3283 - accuracy: 0.9093 - val_loss: 0.3210 - val_accuracy: 0.9100
Epoch 20/100
844/844 - 2s - loss: 0.3225 - accuracy: 0.9107 - val_loss: 0.3151 - val_accuracy: 0.9125
Epoch 21/100
844/844 - 2s - loss: 0.3171 - accuracy: 0.9121 - val_loss: 0.3099 - val_accuracy: 0.9132
Epoch 22/100
844/844 - 2s - loss: 0.3120 - accuracy: 0.9128 - val_loss: 0.3052 - val_accuracy: 0.9130
Epoch 23/100
844/844 - 2s - loss: 0.3073 - accuracy: 0.9146 - val_loss: 0.3005 - val_accuracy: 0.9162
Epoch 24/100
844/844 - 2s - loss: 0.3028 - accuracy: 0.9156 - val_loss: 0.2963 - val_accuracy: 0.9152
Epoch 25/100
844/844 - 2s - loss: 0.2986 - accuracy: 0.9168 - val_loss: 0.2926 - val_accuracy: 0.9168
Epoch 26/100
844/844 - 2s - loss: 0.2946 - accuracy: 0.9178 - val_loss: 0.2884 - val_accuracy: 0.9185
Epoch 27/100
844/844 - 2s - loss: 0.2908 - accuracy: 0.9185 - val_loss: 0.2847 - val_accuracy: 0.9195
Epoch 28/100
844/844 - 2s - loss: 0.2871 - accuracy: 0.9192 - val_loss: 0.2816 - val_accuracy: 0.9202
Epoch 29/100
844/844 - 2s - loss: 0.2837 - accuracy: 0.9205 - val_loss: 0.2784 - val_accuracy: 0.9220
Epoch 30/100
844/844 - 2s - loss: 0.2804 - accuracy: 0.9214 - val_loss: 0.2746 - val_accuracy: 0.9228
Epoch 31/100
844/844 - 2s - loss: 0.2771 - accuracy: 0.9224 - val_loss: 0.2718 - val_accuracy: 0.9240
Epoch 32/100
844/844 - 2s - loss: 0.2740 - accuracy: 0.9233 - val_loss: 0.2689 - val_accuracy: 0.9237
Epoch 33/100
844/844 - 2s - loss: 0.2710 - accuracy: 0.9237 - val_loss: 0.2662 - val_accuracy: 0.9240
Epoch 34/100
844/844 - 2s - loss: 0.2681 - accuracy: 0.9245 - val_loss: 0.2632 - val_accuracy: 0.9252
Epoch 35/100
844/844 - 2s - loss: 0.2652 - accuracy: 0.9250 - val_loss: 0.2607 - val_accuracy: 0.9262
Epoch 36/100
844/844 - 2s - loss: 0.2625 - accuracy: 0.9257 - val_loss: 0.2578 - val_accuracy: 0.9260
Epoch 37/100
844/844 - 2s - loss: 0.2598 - accuracy: 0.9266 - val_loss: 0.2552 - val_accuracy: 0.9263
Epoch 38/100
844/844 - 2s - loss: 0.2573 - accuracy: 0.9275 - val_loss: 0.2526 - val_accuracy: 0.9263
Epoch 39/100
844/844 - 2s - loss: 0.2547 - accuracy: 0.9279 - val_loss: 0.2505 - val_accuracy: 0.9278
Epoch 40/100
844/844 - 2s - loss: 0.2522 - accuracy: 0.9288 - val_loss: 0.2484 - val_accuracy: 0.9283
Epoch 41/100
844/844 - 2s - loss: 0.2499 - accuracy: 0.9295 - val_loss: 0.2462 - val_accuracy: 0.9285
Epoch 42/100
844/844 - 2s - loss: 0.2476 - accuracy: 0.9298 - val_loss: 0.2440 - val_accuracy: 0.9295
Epoch 43/100
844/844 - 2s - loss: 0.2452 - accuracy: 0.9311 - val_loss: 0.2418 - val_accuracy: 0.9295
Epoch 44/100
844/844 - 2s - loss: 0.2431 - accuracy: 0.9318 - val_loss: 0.2394 - val_accuracy: 0.9305
Epoch 45/100
844/844 - 2s - loss: 0.2409 - accuracy: 0.9322 - val_loss: 0.2375 - val_accuracy: 0.9317
Epoch 46/100
844/844 - 2s - loss: 0.2387 - accuracy: 0.9328 - val_loss: 0.2360 - val_accuracy: 0.9317
Epoch 47/100
844/844 - 2s - loss: 0.2366 - accuracy: 0.9336 - val_loss: 0.2338 - val_accuracy: 0.9330
Epoch 48/100
844/844 - 2s - loss: 0.2346 - accuracy: 0.9340 - val_loss: 0.2321 - val_accuracy: 0.9328
Epoch 49/100
844/844 - 2s - loss: 0.2326 - accuracy: 0.9349 - val_loss: 0.2299 - val_accuracy: 0.9340
Epoch 50/100
844/844 - 2s - loss: 0.2306 - accuracy: 0.9355 - val_loss: 0.2278 - val_accuracy: 0.9340
Epoch 51/100
844/844 - 2s - loss: 0.2287 - accuracy: 0.9357 - val_loss: 0.2264 - val_accuracy: 0.9350
Epoch 52/100
844/844 - 2s - loss: 0.2268 - accuracy: 0.9363 - val_loss: 0.2246 - val_accuracy: 0.9360
Epoch 53/100
844/844 - 2s - loss: 0.2249 - accuracy: 0.9369 - val_loss: 0.2229 - val_accuracy: 0.9362
Epoch 54/100
844/844 - 2s - loss: 0.2231 - accuracy: 0.9374 - val_loss: 0.2211 - val_accuracy: 0.9367
Epoch 55/100
844/844 - 2s - loss: 0.2213 - accuracy: 0.9381 - val_loss: 0.2196 - val_accuracy: 0.9377
Epoch 56/100
844/844 - 2s - loss: 0.2196 - accuracy: 0.9384 - val_loss: 0.2182 - val_accuracy: 0.9370
Epoch 57/100
844/844 - 2s - loss: 0.2178 - accuracy: 0.9391 - val_loss: 0.2163 - val_accuracy: 0.9392
Epoch 58/100
844/844 - 2s - loss: 0.2161 - accuracy: 0.9391 - val_loss: 0.2157 - val_accuracy: 0.9388
Epoch 59/100
844/844 - 2s - loss: 0.2144 - accuracy: 0.9398 - val_loss: 0.2134 - val_accuracy: 0.9393
Epoch 60/100
844/844 - 2s - loss: 0.2128 - accuracy: 0.9404 - val_loss: 0.2117 - val_accuracy: 0.9395
Epoch 61/100
844/844 - 2s - loss: 0.2111 - accuracy: 0.9406 - val_loss: 0.2104 - val_accuracy: 0.9405
Epoch 62/100
844/844 - 2s - loss: 0.2096 - accuracy: 0.9412 - val_loss: 0.2086 - val_accuracy: 0.9407
Epoch 63/100
844/844 - 2s - loss: 0.2080 - accuracy: 0.9414 - val_loss: 0.2071 - val_accuracy: 0.9413
Epoch 64/100
844/844 - 2s - loss: 0.2064 - accuracy: 0.9419 - val_loss: 0.2060 - val_accuracy: 0.9413
Epoch 65/100
844/844 - 2s - loss: 0.2049 - accuracy: 0.9422 - val_loss: 0.2043 - val_accuracy: 0.9415
Epoch 66/100
844/844 - 2s - loss: 0.2034 - accuracy: 0.9429 - val_loss: 0.2031 - val_accuracy: 0.9415
Epoch 67/100
844/844 - 2s - loss: 0.2019 - accuracy: 0.9433 - val_loss: 0.2018 - val_accuracy: 0.9420
Epoch 68/100
844/844 - 2s - loss: 0.2005 - accuracy: 0.9438 - val_loss: 0.2004 - val_accuracy: 0.9417
Epoch 69/100
844/844 - 2s - loss: 0.1991 - accuracy: 0.9439 - val_loss: 0.1992 - val_accuracy: 0.9417
Epoch 70/100
844/844 - 2s - loss: 0.1976 - accuracy: 0.9447 - val_loss: 0.1978 - val_accuracy: 0.9418
Epoch 71/100
844/844 - 2s - loss: 0.1962 - accuracy: 0.9451 - val_loss: 0.1970 - val_accuracy: 0.9425
Epoch 72/100
844/844 - 2s - loss: 0.1948 - accuracy: 0.9453 - val_loss: 0.1957 - val_accuracy: 0.9432
Epoch 73/100
844/844 - 2s - loss: 0.1935 - accuracy: 0.9456 - val_loss: 0.1947 - val_accuracy: 0.9430
Epoch 74/100
844/844 - 2s - loss: 0.1921 - accuracy: 0.9461 - val_loss: 0.1925 - val_accuracy: 0.9435
Epoch 75/100
844/844 - 2s - loss: 0.1908 - accuracy: 0.9468 - val_loss: 0.1917 - val_accuracy: 0.9437
Epoch 76/100
844/844 - 2s - loss: 0.1896 - accuracy: 0.9468 - val_loss: 0.1907 - val_accuracy: 0.9445
Epoch 77/100
844/844 - 2s - loss: 0.1883 - accuracy: 0.9472 - val_loss: 0.1893 - val_accuracy: 0.9442
Epoch 78/100
844/844 - 2s - loss: 0.1870 - accuracy: 0.9476 - val_loss: 0.1880 - val_accuracy: 0.9447
Epoch 79/100
844/844 - 2s - loss: 0.1857 - accuracy: 0.9482 - val_loss: 0.1869 - val_accuracy: 0.9448
Epoch 80/100
844/844 - 2s - loss: 0.1845 - accuracy: 0.9485 - val_loss: 0.1859 - val_accuracy: 0.9467
Epoch 81/100
844/844 - 2s - loss: 0.1833 - accuracy: 0.9486 - val_loss: 0.1850 - val_accuracy: 0.9465
Epoch 82/100
844/844 - 2s - loss: 0.1821 - accuracy: 0.9489 - val_loss: 0.1837 - val_accuracy: 0.9463
Epoch 83/100
844/844 - 2s - loss: 0.1809 - accuracy: 0.9493 - val_loss: 0.1827 - val_accuracy: 0.9472
Epoch 84/100
844/844 - 2s - loss: 0.1797 - accuracy: 0.9493 - val_loss: 0.1813 - val_accuracy: 0.9475
Epoch 85/100
844/844 - 2s - loss: 0.1786 - accuracy: 0.9497 - val_loss: 0.1808 - val_accuracy: 0.9477
Epoch 86/100
844/844 - 2s - loss: 0.1774 - accuracy: 0.9505 - val_loss: 0.1800 - val_accuracy: 0.9478
Epoch 87/100
844/844 - 2s - loss: 0.1763 - accuracy: 0.9503 - val_loss: 0.1787 - val_accuracy: 0.9483
Epoch 88/100
844/844 - 2s - loss: 0.1751 - accuracy: 0.9509 - val_loss: 0.1775 - val_accuracy: 0.9490
Epoch 89/100
844/844 - 2s - loss: 0.1740 - accuracy: 0.9514 - val_loss: 0.1769 - val_accuracy: 0.9488
Epoch 90/100
844/844 - 2s - loss: 0.1730 - accuracy: 0.9517 - val_loss: 0.1762 - val_accuracy: 0.9497
Epoch 91/100
844/844 - 2s - loss: 0.1718 - accuracy: 0.9519 - val_loss: 0.1754 - val_accuracy: 0.9497
Epoch 92/100
844/844 - 2s - loss: 0.1708 - accuracy: 0.9518 - val_loss: 0.1741 - val_accuracy: 0.9498
Epoch 93/100
844/844 - 2s - loss: 0.1697 - accuracy: 0.9526 - val_loss: 0.1734 - val_accuracy: 0.9503
Epoch 94/100
844/844 - 2s - loss: 0.1687 - accuracy: 0.9526 - val_loss: 0.1725 - val_accuracy: 0.9510
Epoch 95/100
844/844 - 2s - loss: 0.1676 - accuracy: 0.9531 - val_loss: 0.1720 - val_accuracy: 0.9508
Epoch 96/100
844/844 - 2s - loss: 0.1667 - accuracy: 0.9531 - val_loss: 0.1704 - val_accuracy: 0.9518
Epoch 97/100
844/844 - 2s - loss: 0.1655 - accuracy: 0.9532 - val_loss: 0.1694 - val_accuracy: 0.9512
Epoch 98/100
844/844 - 2s - loss: 0.1646 - accuracy: 0.9541 - val_loss: 0.1690 - val_accuracy: 0.9528
Epoch 99/100
844/844 - 2s - loss: 0.1636 - accuracy: 0.9542 - val_loss: 0.1678 - val_accuracy: 0.9527
Epoch 100/100
844/844 - 2s - loss: 0.1626 - accuracy: 0.9544 - val_loss: 0.1669 - val_accuracy: 0.9517

```


```python
with tf.device(device_name = device_name):
    MLP_model.evaluate(x_test, y_test, batch_size = BATCH_SIZE, verbose = 2)
```

```console
157/157 - 0s - loss: 0.1643 - accuracy: 0.9534

```


```python
show_model_output(x_test[0], MLP_model)
```




```console
<tf.Tensor: shape=(), dtype=int64, numpy=7>
```




![png](MNIST_Trials_files/MNIST_Trials_47_1.png)


### CNN


```python
x_train, x_val, x_test = np.expand_dims(x_train, -1), np.expand_dims(x_val, -1), np.expand_dims(x_test, -1)
```


```python
x_train.shape, x_val.shape, x_test.shape
```




```console
((54000, 28, 28, 1), (6000, 28, 28, 1), (10000, 28, 28, 1))
```




```python
CNN_model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation = "relu"),
    keras.layers.Dense(128, activation = "relu"),
    keras.layers.Dense(10, activation = "softmax"),
])

CNN_model.summary()
CNN_model.compile(loss = loss, optimizer = optimizer, metrics = metrics)

with tf.device(device_name = device_name):
    print("training start!")
    hist = CNN_model.fit(x_train, y_train, batch_size = BATCH_SIZE, epochs = EPOCHS, shuffle = True, verbose = 2, validation_data=(x_val, y_val))
    print("evaluation start!")
    CNN_model.evaluate(x_test, y_test, batch_size = BATCH_SIZE, verbose = 2)

#show_model_output(x_test[0], CNN_model)
```

```console
Model: "sequential_7"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_3 (Conv2D)            (None, 26, 26, 32)        320       
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 13, 13, 32)        0         
_________________________________________________________________
flatten_7 (Flatten)          (None, 5408)              0         
_________________________________________________________________
dense_19 (Dense)             (None, 256)               1384704   
_________________________________________________________________
dense_20 (Dense)             (None, 128)               32896     
_________________________________________________________________
dense_21 (Dense)             (None, 10)                1290      
=================================================================
Total params: 1,419,210
Trainable params: 1,419,210
Non-trainable params: 0
_________________________________________________________________
training start!
Epoch 1/100
844/844 - 30s - loss: 2.1322 - accuracy: 0.4303 - val_loss: 1.8467 - val_accuracy: 0.6618
Epoch 2/100
844/844 - 2s - loss: 1.3173 - accuracy: 0.7521 - val_loss: 0.8418 - val_accuracy: 0.8143
Epoch 3/100
844/844 - 2s - loss: 0.6377 - accuracy: 0.8448 - val_loss: 0.5103 - val_accuracy: 0.8675
Epoch 4/100
844/844 - 2s - loss: 0.4542 - accuracy: 0.8782 - val_loss: 0.4086 - val_accuracy: 0.8892
Epoch 5/100
844/844 - 2s - loss: 0.3849 - accuracy: 0.8925 - val_loss: 0.3585 - val_accuracy: 0.9032
Epoch 6/100
844/844 - 2s - loss: 0.3481 - accuracy: 0.9006 - val_loss: 0.3274 - val_accuracy: 0.9098
Epoch 7/100
844/844 - 2s - loss: 0.3240 - accuracy: 0.9067 - val_loss: 0.3074 - val_accuracy: 0.9172
Epoch 8/100
844/844 - 2s - loss: 0.3062 - accuracy: 0.9104 - val_loss: 0.2919 - val_accuracy: 0.9197
Epoch 9/100
844/844 - 2s - loss: 0.2924 - accuracy: 0.9145 - val_loss: 0.2778 - val_accuracy: 0.9227
Epoch 10/100
844/844 - 2s - loss: 0.2806 - accuracy: 0.9179 - val_loss: 0.2672 - val_accuracy: 0.9273
Epoch 11/100
844/844 - 2s - loss: 0.2705 - accuracy: 0.9201 - val_loss: 0.2575 - val_accuracy: 0.9292
Epoch 12/100
844/844 - 2s - loss: 0.2612 - accuracy: 0.9234 - val_loss: 0.2497 - val_accuracy: 0.9318
Epoch 13/100
844/844 - 2s - loss: 0.2531 - accuracy: 0.9255 - val_loss: 0.2408 - val_accuracy: 0.9352
Epoch 14/100
844/844 - 2s - loss: 0.2453 - accuracy: 0.9281 - val_loss: 0.2347 - val_accuracy: 0.9358
Epoch 15/100
844/844 - 2s - loss: 0.2383 - accuracy: 0.9301 - val_loss: 0.2270 - val_accuracy: 0.9372
Epoch 16/100
844/844 - 2s - loss: 0.2314 - accuracy: 0.9319 - val_loss: 0.2203 - val_accuracy: 0.9395
Epoch 17/100
844/844 - 2s - loss: 0.2253 - accuracy: 0.9336 - val_loss: 0.2142 - val_accuracy: 0.9397
Epoch 18/100
844/844 - 2s - loss: 0.2192 - accuracy: 0.9356 - val_loss: 0.2091 - val_accuracy: 0.9412
Epoch 19/100
844/844 - 2s - loss: 0.2138 - accuracy: 0.9374 - val_loss: 0.2034 - val_accuracy: 0.9445
Epoch 20/100
844/844 - 2s - loss: 0.2084 - accuracy: 0.9392 - val_loss: 0.1995 - val_accuracy: 0.9448
Epoch 21/100
844/844 - 2s - loss: 0.2035 - accuracy: 0.9399 - val_loss: 0.1937 - val_accuracy: 0.9460
Epoch 22/100
844/844 - 2s - loss: 0.1988 - accuracy: 0.9419 - val_loss: 0.1907 - val_accuracy: 0.9458
Epoch 23/100
844/844 - 2s - loss: 0.1941 - accuracy: 0.9433 - val_loss: 0.1857 - val_accuracy: 0.9485
Epoch 24/100
844/844 - 2s - loss: 0.1899 - accuracy: 0.9444 - val_loss: 0.1816 - val_accuracy: 0.9492
Epoch 25/100
844/844 - 2s - loss: 0.1857 - accuracy: 0.9454 - val_loss: 0.1771 - val_accuracy: 0.9510
Epoch 26/100
844/844 - 2s - loss: 0.1818 - accuracy: 0.9462 - val_loss: 0.1749 - val_accuracy: 0.9515
Epoch 27/100
844/844 - 2s - loss: 0.1778 - accuracy: 0.9475 - val_loss: 0.1708 - val_accuracy: 0.9532
Epoch 28/100
844/844 - 2s - loss: 0.1742 - accuracy: 0.9489 - val_loss: 0.1679 - val_accuracy: 0.9528
Epoch 29/100
844/844 - 2s - loss: 0.1708 - accuracy: 0.9495 - val_loss: 0.1650 - val_accuracy: 0.9543
Epoch 30/100
844/844 - 2s - loss: 0.1673 - accuracy: 0.9505 - val_loss: 0.1618 - val_accuracy: 0.9545
Epoch 31/100
844/844 - 2s - loss: 0.1641 - accuracy: 0.9520 - val_loss: 0.1605 - val_accuracy: 0.9555
Epoch 32/100
844/844 - 2s - loss: 0.1608 - accuracy: 0.9525 - val_loss: 0.1560 - val_accuracy: 0.9563
Epoch 33/100
844/844 - 2s - loss: 0.1578 - accuracy: 0.9531 - val_loss: 0.1537 - val_accuracy: 0.9560
Epoch 34/100
844/844 - 2s - loss: 0.1549 - accuracy: 0.9539 - val_loss: 0.1511 - val_accuracy: 0.9573
Epoch 35/100
844/844 - 2s - loss: 0.1522 - accuracy: 0.9547 - val_loss: 0.1488 - val_accuracy: 0.9573
Epoch 36/100
844/844 - 2s - loss: 0.1496 - accuracy: 0.9555 - val_loss: 0.1491 - val_accuracy: 0.9578
Epoch 37/100
844/844 - 2s - loss: 0.1468 - accuracy: 0.9572 - val_loss: 0.1444 - val_accuracy: 0.9585
Epoch 38/100
844/844 - 2s - loss: 0.1445 - accuracy: 0.9572 - val_loss: 0.1420 - val_accuracy: 0.9593
Epoch 39/100
844/844 - 2s - loss: 0.1418 - accuracy: 0.9579 - val_loss: 0.1409 - val_accuracy: 0.9593
Epoch 40/100
844/844 - 2s - loss: 0.1395 - accuracy: 0.9591 - val_loss: 0.1382 - val_accuracy: 0.9608
Epoch 41/100
844/844 - 2s - loss: 0.1373 - accuracy: 0.9596 - val_loss: 0.1375 - val_accuracy: 0.9617
Epoch 42/100
844/844 - 2s - loss: 0.1351 - accuracy: 0.9603 - val_loss: 0.1344 - val_accuracy: 0.9622
Epoch 43/100
844/844 - 2s - loss: 0.1329 - accuracy: 0.9610 - val_loss: 0.1344 - val_accuracy: 0.9635
Epoch 44/100
844/844 - 2s - loss: 0.1309 - accuracy: 0.9615 - val_loss: 0.1300 - val_accuracy: 0.9627
Epoch 45/100
844/844 - 2s - loss: 0.1288 - accuracy: 0.9626 - val_loss: 0.1283 - val_accuracy: 0.9630
Epoch 46/100
844/844 - 2s - loss: 0.1268 - accuracy: 0.9628 - val_loss: 0.1280 - val_accuracy: 0.9627
Epoch 47/100
844/844 - 2s - loss: 0.1249 - accuracy: 0.9637 - val_loss: 0.1255 - val_accuracy: 0.9647
Epoch 48/100
844/844 - 2s - loss: 0.1230 - accuracy: 0.9643 - val_loss: 0.1257 - val_accuracy: 0.9642
Epoch 49/100
844/844 - 2s - loss: 0.1213 - accuracy: 0.9647 - val_loss: 0.1225 - val_accuracy: 0.9643
Epoch 50/100
844/844 - 2s - loss: 0.1195 - accuracy: 0.9653 - val_loss: 0.1223 - val_accuracy: 0.9658
Epoch 51/100
844/844 - 2s - loss: 0.1178 - accuracy: 0.9654 - val_loss: 0.1189 - val_accuracy: 0.9652
Epoch 52/100
844/844 - 2s - loss: 0.1160 - accuracy: 0.9664 - val_loss: 0.1210 - val_accuracy: 0.9648
Epoch 53/100
844/844 - 2s - loss: 0.1143 - accuracy: 0.9670 - val_loss: 0.1184 - val_accuracy: 0.9655
Epoch 54/100
844/844 - 2s - loss: 0.1126 - accuracy: 0.9676 - val_loss: 0.1166 - val_accuracy: 0.9663
Epoch 55/100
844/844 - 2s - loss: 0.1112 - accuracy: 0.9680 - val_loss: 0.1161 - val_accuracy: 0.9673
Epoch 56/100
844/844 - 2s - loss: 0.1096 - accuracy: 0.9683 - val_loss: 0.1140 - val_accuracy: 0.9657
Epoch 57/100
844/844 - 2s - loss: 0.1082 - accuracy: 0.9688 - val_loss: 0.1144 - val_accuracy: 0.9665
Epoch 58/100
844/844 - 2s - loss: 0.1067 - accuracy: 0.9694 - val_loss: 0.1124 - val_accuracy: 0.9670
Epoch 59/100
844/844 - 2s - loss: 0.1053 - accuracy: 0.9697 - val_loss: 0.1103 - val_accuracy: 0.9675
Epoch 60/100
844/844 - 2s - loss: 0.1039 - accuracy: 0.9703 - val_loss: 0.1100 - val_accuracy: 0.9685
Epoch 61/100
844/844 - 2s - loss: 0.1026 - accuracy: 0.9706 - val_loss: 0.1087 - val_accuracy: 0.9687
Epoch 62/100
844/844 - 2s - loss: 0.1012 - accuracy: 0.9708 - val_loss: 0.1076 - val_accuracy: 0.9670
Epoch 63/100
844/844 - 2s - loss: 0.1000 - accuracy: 0.9710 - val_loss: 0.1073 - val_accuracy: 0.9692
Epoch 64/100
844/844 - 2s - loss: 0.0987 - accuracy: 0.9717 - val_loss: 0.1044 - val_accuracy: 0.9692
Epoch 65/100
844/844 - 2s - loss: 0.0975 - accuracy: 0.9719 - val_loss: 0.1043 - val_accuracy: 0.9693
Epoch 66/100
844/844 - 2s - loss: 0.0962 - accuracy: 0.9721 - val_loss: 0.1046 - val_accuracy: 0.9695
Epoch 67/100
844/844 - 2s - loss: 0.0952 - accuracy: 0.9729 - val_loss: 0.1024 - val_accuracy: 0.9698
Epoch 68/100
844/844 - 2s - loss: 0.0940 - accuracy: 0.9730 - val_loss: 0.1005 - val_accuracy: 0.9693
Epoch 69/100
844/844 - 2s - loss: 0.0928 - accuracy: 0.9732 - val_loss: 0.1012 - val_accuracy: 0.9707
Epoch 70/100
844/844 - 2s - loss: 0.0916 - accuracy: 0.9739 - val_loss: 0.0998 - val_accuracy: 0.9707
Epoch 71/100
844/844 - 2s - loss: 0.0906 - accuracy: 0.9738 - val_loss: 0.0993 - val_accuracy: 0.9705
Epoch 72/100
844/844 - 2s - loss: 0.0895 - accuracy: 0.9740 - val_loss: 0.0986 - val_accuracy: 0.9717
Epoch 73/100
844/844 - 2s - loss: 0.0886 - accuracy: 0.9744 - val_loss: 0.0978 - val_accuracy: 0.9717
Epoch 74/100
844/844 - 2s - loss: 0.0875 - accuracy: 0.9747 - val_loss: 0.0968 - val_accuracy: 0.9723
Epoch 75/100
844/844 - 2s - loss: 0.0864 - accuracy: 0.9753 - val_loss: 0.0958 - val_accuracy: 0.9718
Epoch 76/100
844/844 - 2s - loss: 0.0857 - accuracy: 0.9753 - val_loss: 0.0956 - val_accuracy: 0.9718
Epoch 77/100
844/844 - 2s - loss: 0.0846 - accuracy: 0.9760 - val_loss: 0.0952 - val_accuracy: 0.9717
Epoch 78/100
844/844 - 2s - loss: 0.0837 - accuracy: 0.9761 - val_loss: 0.0936 - val_accuracy: 0.9720
Epoch 79/100
844/844 - 2s - loss: 0.0826 - accuracy: 0.9765 - val_loss: 0.0938 - val_accuracy: 0.9728
Epoch 80/100
844/844 - 2s - loss: 0.0817 - accuracy: 0.9769 - val_loss: 0.0928 - val_accuracy: 0.9715
Epoch 81/100
844/844 - 2s - loss: 0.0809 - accuracy: 0.9767 - val_loss: 0.0928 - val_accuracy: 0.9727
Epoch 82/100
844/844 - 2s - loss: 0.0800 - accuracy: 0.9772 - val_loss: 0.0915 - val_accuracy: 0.9728
Epoch 83/100
844/844 - 2s - loss: 0.0791 - accuracy: 0.9777 - val_loss: 0.0903 - val_accuracy: 0.9737
Epoch 84/100
844/844 - 2s - loss: 0.0782 - accuracy: 0.9778 - val_loss: 0.0906 - val_accuracy: 0.9733
Epoch 85/100
844/844 - 2s - loss: 0.0774 - accuracy: 0.9777 - val_loss: 0.0878 - val_accuracy: 0.9757
Epoch 86/100
844/844 - 2s - loss: 0.0767 - accuracy: 0.9779 - val_loss: 0.0895 - val_accuracy: 0.9742
Epoch 87/100
844/844 - 2s - loss: 0.0757 - accuracy: 0.9782 - val_loss: 0.0881 - val_accuracy: 0.9742
Epoch 88/100
844/844 - 2s - loss: 0.0750 - accuracy: 0.9784 - val_loss: 0.0875 - val_accuracy: 0.9753
Epoch 89/100
844/844 - 2s - loss: 0.0742 - accuracy: 0.9788 - val_loss: 0.0891 - val_accuracy: 0.9750
Epoch 90/100
844/844 - 2s - loss: 0.0735 - accuracy: 0.9791 - val_loss: 0.0874 - val_accuracy: 0.9740
Epoch 91/100
844/844 - 2s - loss: 0.0728 - accuracy: 0.9794 - val_loss: 0.0856 - val_accuracy: 0.9758
Epoch 92/100
844/844 - 2s - loss: 0.0720 - accuracy: 0.9791 - val_loss: 0.0880 - val_accuracy: 0.9745
Epoch 93/100
844/844 - 2s - loss: 0.0713 - accuracy: 0.9795 - val_loss: 0.0862 - val_accuracy: 0.9750
Epoch 94/100
844/844 - 2s - loss: 0.0706 - accuracy: 0.9799 - val_loss: 0.0859 - val_accuracy: 0.9758
Epoch 95/100
844/844 - 2s - loss: 0.0698 - accuracy: 0.9801 - val_loss: 0.0851 - val_accuracy: 0.9763
Epoch 96/100
844/844 - 2s - loss: 0.0692 - accuracy: 0.9803 - val_loss: 0.0834 - val_accuracy: 0.9763
Epoch 97/100
844/844 - 2s - loss: 0.0686 - accuracy: 0.9803 - val_loss: 0.0833 - val_accuracy: 0.9770
Epoch 98/100
844/844 - 2s - loss: 0.0678 - accuracy: 0.9803 - val_loss: 0.0849 - val_accuracy: 0.9735
Epoch 99/100
844/844 - 2s - loss: 0.0670 - accuracy: 0.9812 - val_loss: 0.0814 - val_accuracy: 0.9772
Epoch 100/100
844/844 - 2s - loss: 0.0666 - accuracy: 0.9810 - val_loss: 0.0834 - val_accuracy: 0.9753
evaluation start!
157/157 - 0s - loss: 0.0778 - accuracy: 0.9767

```


```console
---------------------------------------------------------------------------
```

```console
TypeError                                 Traceback (most recent call last)
```

```console
<ipython-input-55-bafad150fd19> in <module>()
     17     CNN_model.evaluate(x_test, y_test, batch_size = BATCH_SIZE, verbose = 2)
     18 
---> 19 show_model_output(x_test[0], CNN_model)

```

```console
<ipython-input-41-13d295f90d77> in show_model_output(data, model)
      1 def show_model_output(data, model):
----> 2     plt.imshow(data)
      3     expanded_data = tf.expand_dims(data, axis = 0)
      4     model_output = model(expanded_data)
      5     return tf.math.argmax(model_output, axis = 1)[0]

```

```console
/usr/local/lib/python3.7/dist-packages/matplotlib/pyplot.py in imshow(X, cmap, norm, aspect, interpolation, alpha, vmin, vmax, origin, extent, shape, filternorm, filterrad, imlim, resample, url, data, **kwargs)
   2649         filternorm=filternorm, filterrad=filterrad, imlim=imlim,
   2650         resample=resample, url=url, **({"data": data} if data is not
-> 2651         None else {}), **kwargs)
   2652     sci(__ret)
   2653     return __ret

```

```console
/usr/local/lib/python3.7/dist-packages/matplotlib/__init__.py in inner(ax, data, *args, **kwargs)
   1563     def inner(ax, *args, data=None, **kwargs):
   1564         if data is None:
-> 1565             return func(ax, *map(sanitize_sequence, args), **kwargs)
   1566 
   1567         bound = new_sig.bind(ax, *args, **kwargs)

```

```console
/usr/local/lib/python3.7/dist-packages/matplotlib/cbook/deprecation.py in wrapper(*args, **kwargs)
    356                 f"%(removal)s.  If any parameter follows {name!r}, they "
    357                 f"should be pass as keyword, not positionally.")
--> 358         return func(*args, **kwargs)
    359 
    360     return wrapper

```

```console
/usr/local/lib/python3.7/dist-packages/matplotlib/cbook/deprecation.py in wrapper(*args, **kwargs)
    356                 f"%(removal)s.  If any parameter follows {name!r}, they "
    357                 f"should be pass as keyword, not positionally.")
--> 358         return func(*args, **kwargs)
    359 
    360     return wrapper

```

```console
/usr/local/lib/python3.7/dist-packages/matplotlib/axes/_axes.py in imshow(self, X, cmap, norm, aspect, interpolation, alpha, vmin, vmax, origin, extent, shape, filternorm, filterrad, imlim, resample, url, **kwargs)
   5624                               resample=resample, **kwargs)
   5625 
-> 5626         im.set_data(X)
   5627         im.set_alpha(alpha)
   5628         if im.get_clip_path() is None:

```

```console
/usr/local/lib/python3.7/dist-packages/matplotlib/image.py in set_data(self, A)
    697                 or self._A.ndim == 3 and self._A.shape[-1] in [3, 4]):
    698             raise TypeError("Invalid shape {} for image data"
--> 699                             .format(self._A.shape))
    700 
    701         if self._A.ndim == 3:

```

```console
TypeError: Invalid shape (28, 28, 1) for image data
```



![png](MNIST_Trials_files/MNIST_Trials_51_2.png)


### Deeper CNN


```python
Deeper_CNN_model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Flatten(),
    keras.layers.Dense(256, activation = "relu"),
    keras.layers.Dense(128, activation = "relu"),
    keras.layers.Dense(10, activation = "softmax"),
])

Deeper_CNN_model.summary()
Deeper_CNN_model.compile(loss = loss, optimizer = optimizer, metrics = metrics)

with tf.device(device_name = device_name):
    print("training start!")
    hist = Deeper_CNN_model.fit(x_train, y_train, batch_size = BATCH_SIZE, epochs = EPOCHS, shuffle = True, verbose = 2, validation_data=(x_val, y_val))
    print("evaluation start!")
    Deeper_CNN_model.evaluate(x_test, y_test, batch_size = BATCH_SIZE, verbose = 2)

#show_model_output(x_test[0], Deeper_CNN_model)
```

```console
Model: "sequential_8"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_4 (Conv2D)            (None, 26, 26, 32)        320       
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 13, 13, 32)        0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 11, 11, 64)        18496     
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 5, 5, 64)          0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 3, 3, 128)         73856     
_________________________________________________________________
max_pooling2d_6 (MaxPooling2 (None, 1, 1, 128)         0         
_________________________________________________________________
flatten_8 (Flatten)          (None, 128)               0         
_________________________________________________________________
dense_22 (Dense)             (None, 256)               33024     
_________________________________________________________________
dense_23 (Dense)             (None, 128)               32896     
_________________________________________________________________
dense_24 (Dense)             (None, 10)                1290      
=================================================================
Total params: 159,882
Trainable params: 159,882
Non-trainable params: 0
_________________________________________________________________
training start!
Epoch 1/100
844/844 - 3s - loss: 2.2933 - accuracy: 0.1166 - val_loss: 2.2832 - val_accuracy: 0.1647
Epoch 2/100
844/844 - 3s - loss: 2.2726 - accuracy: 0.2655 - val_loss: 2.2608 - val_accuracy: 0.3445
Epoch 3/100
844/844 - 3s - loss: 2.2423 - accuracy: 0.4044 - val_loss: 2.2188 - val_accuracy: 0.4573
Epoch 4/100
844/844 - 3s - loss: 2.1759 - accuracy: 0.5106 - val_loss: 2.1148 - val_accuracy: 0.5583
Epoch 5/100
844/844 - 2s - loss: 1.9721 - accuracy: 0.6097 - val_loss: 1.7511 - val_accuracy: 0.6535
Epoch 6/100
844/844 - 3s - loss: 1.3956 - accuracy: 0.7049 - val_loss: 1.0557 - val_accuracy: 0.7668
Epoch 7/100
844/844 - 3s - loss: 0.8295 - accuracy: 0.8006 - val_loss: 0.6508 - val_accuracy: 0.8367
Epoch 8/100
844/844 - 3s - loss: 0.5657 - accuracy: 0.8484 - val_loss: 0.4886 - val_accuracy: 0.8652
Epoch 9/100
844/844 - 3s - loss: 0.4491 - accuracy: 0.8716 - val_loss: 0.3975 - val_accuracy: 0.8825
Epoch 10/100
844/844 - 3s - loss: 0.3839 - accuracy: 0.8876 - val_loss: 0.3456 - val_accuracy: 0.8955
Epoch 11/100
844/844 - 3s - loss: 0.3396 - accuracy: 0.8994 - val_loss: 0.3083 - val_accuracy: 0.9085
Epoch 12/100
844/844 - 3s - loss: 0.3069 - accuracy: 0.9088 - val_loss: 0.2801 - val_accuracy: 0.9157
Epoch 13/100
844/844 - 3s - loss: 0.2807 - accuracy: 0.9165 - val_loss: 0.2632 - val_accuracy: 0.9215
Epoch 14/100
844/844 - 3s - loss: 0.2606 - accuracy: 0.9220 - val_loss: 0.2391 - val_accuracy: 0.9288
Epoch 15/100
844/844 - 3s - loss: 0.2440 - accuracy: 0.9267 - val_loss: 0.2229 - val_accuracy: 0.9315
Epoch 16/100
844/844 - 3s - loss: 0.2296 - accuracy: 0.9306 - val_loss: 0.2158 - val_accuracy: 0.9353
Epoch 17/100
844/844 - 3s - loss: 0.2176 - accuracy: 0.9344 - val_loss: 0.2085 - val_accuracy: 0.9368
Epoch 18/100
844/844 - 3s - loss: 0.2068 - accuracy: 0.9366 - val_loss: 0.1927 - val_accuracy: 0.9412
Epoch 19/100
844/844 - 3s - loss: 0.1980 - accuracy: 0.9391 - val_loss: 0.1859 - val_accuracy: 0.9448
Epoch 20/100
844/844 - 3s - loss: 0.1894 - accuracy: 0.9421 - val_loss: 0.1813 - val_accuracy: 0.9443
Epoch 21/100
844/844 - 3s - loss: 0.1821 - accuracy: 0.9436 - val_loss: 0.1732 - val_accuracy: 0.9473
Epoch 22/100
844/844 - 3s - loss: 0.1754 - accuracy: 0.9463 - val_loss: 0.1684 - val_accuracy: 0.9465
Epoch 23/100
844/844 - 3s - loss: 0.1692 - accuracy: 0.9483 - val_loss: 0.1594 - val_accuracy: 0.9535
Epoch 24/100
844/844 - 3s - loss: 0.1636 - accuracy: 0.9495 - val_loss: 0.1564 - val_accuracy: 0.9545
Epoch 25/100
844/844 - 3s - loss: 0.1582 - accuracy: 0.9515 - val_loss: 0.1568 - val_accuracy: 0.9548
Epoch 26/100
844/844 - 3s - loss: 0.1532 - accuracy: 0.9529 - val_loss: 0.1550 - val_accuracy: 0.9528
Epoch 27/100
844/844 - 3s - loss: 0.1490 - accuracy: 0.9545 - val_loss: 0.1436 - val_accuracy: 0.9585
Epoch 28/100
844/844 - 3s - loss: 0.1446 - accuracy: 0.9559 - val_loss: 0.1482 - val_accuracy: 0.9573
Epoch 29/100
844/844 - 2s - loss: 0.1409 - accuracy: 0.9560 - val_loss: 0.1362 - val_accuracy: 0.9588
Epoch 30/100
844/844 - 2s - loss: 0.1371 - accuracy: 0.9583 - val_loss: 0.1369 - val_accuracy: 0.9605
Epoch 31/100
844/844 - 3s - loss: 0.1336 - accuracy: 0.9589 - val_loss: 0.1374 - val_accuracy: 0.9592
Epoch 32/100
844/844 - 3s - loss: 0.1309 - accuracy: 0.9594 - val_loss: 0.1337 - val_accuracy: 0.9617
Epoch 33/100
844/844 - 3s - loss: 0.1277 - accuracy: 0.9610 - val_loss: 0.1357 - val_accuracy: 0.9593
Epoch 34/100
844/844 - 3s - loss: 0.1248 - accuracy: 0.9621 - val_loss: 0.1243 - val_accuracy: 0.9645
Epoch 35/100
844/844 - 3s - loss: 0.1219 - accuracy: 0.9630 - val_loss: 0.1250 - val_accuracy: 0.9642
Epoch 36/100
844/844 - 3s - loss: 0.1197 - accuracy: 0.9634 - val_loss: 0.1204 - val_accuracy: 0.9657
Epoch 37/100
844/844 - 3s - loss: 0.1174 - accuracy: 0.9644 - val_loss: 0.1194 - val_accuracy: 0.9668
Epoch 38/100
844/844 - 3s - loss: 0.1153 - accuracy: 0.9642 - val_loss: 0.1177 - val_accuracy: 0.9665
Epoch 39/100
844/844 - 3s - loss: 0.1133 - accuracy: 0.9652 - val_loss: 0.1177 - val_accuracy: 0.9670
Epoch 40/100
844/844 - 3s - loss: 0.1108 - accuracy: 0.9660 - val_loss: 0.1107 - val_accuracy: 0.9695
Epoch 41/100
844/844 - 3s - loss: 0.1085 - accuracy: 0.9669 - val_loss: 0.1091 - val_accuracy: 0.9705
Epoch 42/100
844/844 - 3s - loss: 0.1068 - accuracy: 0.9671 - val_loss: 0.1142 - val_accuracy: 0.9652
Epoch 43/100
844/844 - 3s - loss: 0.1050 - accuracy: 0.9677 - val_loss: 0.1069 - val_accuracy: 0.9698
Epoch 44/100
844/844 - 2s - loss: 0.1037 - accuracy: 0.9684 - val_loss: 0.1090 - val_accuracy: 0.9710
Epoch 45/100
844/844 - 3s - loss: 0.1020 - accuracy: 0.9687 - val_loss: 0.1060 - val_accuracy: 0.9687
Epoch 46/100
844/844 - 2s - loss: 0.1006 - accuracy: 0.9695 - val_loss: 0.1042 - val_accuracy: 0.9718
Epoch 47/100
844/844 - 2s - loss: 0.0985 - accuracy: 0.9705 - val_loss: 0.1127 - val_accuracy: 0.9687
Epoch 48/100
844/844 - 3s - loss: 0.0971 - accuracy: 0.9705 - val_loss: 0.1032 - val_accuracy: 0.9712
Epoch 49/100
844/844 - 3s - loss: 0.0959 - accuracy: 0.9708 - val_loss: 0.1014 - val_accuracy: 0.9730
Epoch 50/100
844/844 - 2s - loss: 0.0945 - accuracy: 0.9711 - val_loss: 0.1000 - val_accuracy: 0.9717
Epoch 51/100
844/844 - 3s - loss: 0.0926 - accuracy: 0.9723 - val_loss: 0.0965 - val_accuracy: 0.9733
Epoch 52/100
844/844 - 3s - loss: 0.0922 - accuracy: 0.9720 - val_loss: 0.1012 - val_accuracy: 0.9718
Epoch 53/100
844/844 - 3s - loss: 0.0905 - accuracy: 0.9724 - val_loss: 0.1004 - val_accuracy: 0.9712
Epoch 54/100
844/844 - 2s - loss: 0.0888 - accuracy: 0.9727 - val_loss: 0.1015 - val_accuracy: 0.9715
Epoch 55/100
844/844 - 2s - loss: 0.0877 - accuracy: 0.9730 - val_loss: 0.1098 - val_accuracy: 0.9685
Epoch 56/100
844/844 - 3s - loss: 0.0864 - accuracy: 0.9731 - val_loss: 0.0982 - val_accuracy: 0.9728
Epoch 57/100
844/844 - 3s - loss: 0.0858 - accuracy: 0.9739 - val_loss: 0.0971 - val_accuracy: 0.9717
Epoch 58/100
844/844 - 3s - loss: 0.0849 - accuracy: 0.9738 - val_loss: 0.0979 - val_accuracy: 0.9727
Epoch 59/100
844/844 - 3s - loss: 0.0837 - accuracy: 0.9746 - val_loss: 0.0922 - val_accuracy: 0.9742
Epoch 60/100
844/844 - 3s - loss: 0.0821 - accuracy: 0.9750 - val_loss: 0.0924 - val_accuracy: 0.9735
Epoch 61/100
844/844 - 3s - loss: 0.0815 - accuracy: 0.9753 - val_loss: 0.0904 - val_accuracy: 0.9745
Epoch 62/100
844/844 - 3s - loss: 0.0803 - accuracy: 0.9753 - val_loss: 0.0985 - val_accuracy: 0.9720
Epoch 63/100
844/844 - 3s - loss: 0.0797 - accuracy: 0.9759 - val_loss: 0.0882 - val_accuracy: 0.9755
Epoch 64/100
844/844 - 3s - loss: 0.0793 - accuracy: 0.9756 - val_loss: 0.0895 - val_accuracy: 0.9747
Epoch 65/100
844/844 - 3s - loss: 0.0779 - accuracy: 0.9759 - val_loss: 0.0881 - val_accuracy: 0.9743
Epoch 66/100
844/844 - 3s - loss: 0.0768 - accuracy: 0.9763 - val_loss: 0.0866 - val_accuracy: 0.9758
Epoch 67/100
844/844 - 3s - loss: 0.0756 - accuracy: 0.9772 - val_loss: 0.0897 - val_accuracy: 0.9745
Epoch 68/100
844/844 - 3s - loss: 0.0748 - accuracy: 0.9769 - val_loss: 0.0878 - val_accuracy: 0.9755
Epoch 69/100
844/844 - 2s - loss: 0.0744 - accuracy: 0.9770 - val_loss: 0.0895 - val_accuracy: 0.9740
Epoch 70/100
844/844 - 2s - loss: 0.0735 - accuracy: 0.9771 - val_loss: 0.0876 - val_accuracy: 0.9763
Epoch 71/100
844/844 - 2s - loss: 0.0728 - accuracy: 0.9776 - val_loss: 0.0908 - val_accuracy: 0.9732
Epoch 72/100
844/844 - 3s - loss: 0.0718 - accuracy: 0.9780 - val_loss: 0.0890 - val_accuracy: 0.9753
Epoch 73/100
844/844 - 2s - loss: 0.0717 - accuracy: 0.9780 - val_loss: 0.0863 - val_accuracy: 0.9757
Epoch 74/100
844/844 - 2s - loss: 0.0702 - accuracy: 0.9787 - val_loss: 0.0890 - val_accuracy: 0.9737
Epoch 75/100
844/844 - 2s - loss: 0.0701 - accuracy: 0.9786 - val_loss: 0.0850 - val_accuracy: 0.9750
Epoch 76/100
844/844 - 3s - loss: 0.0695 - accuracy: 0.9787 - val_loss: 0.0884 - val_accuracy: 0.9740
Epoch 77/100
844/844 - 2s - loss: 0.0686 - accuracy: 0.9788 - val_loss: 0.0868 - val_accuracy: 0.9752
Epoch 78/100
844/844 - 3s - loss: 0.0674 - accuracy: 0.9791 - val_loss: 0.0856 - val_accuracy: 0.9763
Epoch 79/100
844/844 - 3s - loss: 0.0669 - accuracy: 0.9793 - val_loss: 0.0897 - val_accuracy: 0.9752
Epoch 80/100
844/844 - 3s - loss: 0.0667 - accuracy: 0.9797 - val_loss: 0.0815 - val_accuracy: 0.9755
Epoch 81/100
844/844 - 2s - loss: 0.0660 - accuracy: 0.9799 - val_loss: 0.0834 - val_accuracy: 0.9760
Epoch 82/100
844/844 - 2s - loss: 0.0654 - accuracy: 0.9797 - val_loss: 0.0822 - val_accuracy: 0.9758
Epoch 83/100
844/844 - 2s - loss: 0.0647 - accuracy: 0.9802 - val_loss: 0.0812 - val_accuracy: 0.9773
Epoch 84/100
844/844 - 3s - loss: 0.0641 - accuracy: 0.9802 - val_loss: 0.0838 - val_accuracy: 0.9755
Epoch 85/100
844/844 - 2s - loss: 0.0633 - accuracy: 0.9805 - val_loss: 0.0781 - val_accuracy: 0.9770
Epoch 86/100
844/844 - 2s - loss: 0.0630 - accuracy: 0.9808 - val_loss: 0.0795 - val_accuracy: 0.9777
Epoch 87/100
844/844 - 3s - loss: 0.0622 - accuracy: 0.9810 - val_loss: 0.0806 - val_accuracy: 0.9772
Epoch 88/100
844/844 - 3s - loss: 0.0616 - accuracy: 0.9806 - val_loss: 0.0801 - val_accuracy: 0.9773
Epoch 89/100
844/844 - 2s - loss: 0.0612 - accuracy: 0.9810 - val_loss: 0.0777 - val_accuracy: 0.9777
Epoch 90/100
844/844 - 2s - loss: 0.0604 - accuracy: 0.9814 - val_loss: 0.0799 - val_accuracy: 0.9777
Epoch 91/100
844/844 - 2s - loss: 0.0602 - accuracy: 0.9815 - val_loss: 0.0814 - val_accuracy: 0.9768
Epoch 92/100
844/844 - 3s - loss: 0.0597 - accuracy: 0.9819 - val_loss: 0.0755 - val_accuracy: 0.9773
Epoch 93/100
844/844 - 2s - loss: 0.0589 - accuracy: 0.9817 - val_loss: 0.0823 - val_accuracy: 0.9770
Epoch 94/100
844/844 - 2s - loss: 0.0585 - accuracy: 0.9817 - val_loss: 0.0807 - val_accuracy: 0.9770
Epoch 95/100
844/844 - 2s - loss: 0.0576 - accuracy: 0.9825 - val_loss: 0.0784 - val_accuracy: 0.9775
Epoch 96/100
844/844 - 3s - loss: 0.0572 - accuracy: 0.9826 - val_loss: 0.0800 - val_accuracy: 0.9760
Epoch 97/100
844/844 - 3s - loss: 0.0566 - accuracy: 0.9824 - val_loss: 0.0790 - val_accuracy: 0.9775
Epoch 98/100
844/844 - 3s - loss: 0.0563 - accuracy: 0.9827 - val_loss: 0.0829 - val_accuracy: 0.9758
Epoch 99/100
844/844 - 3s - loss: 0.0559 - accuracy: 0.9828 - val_loss: 0.0759 - val_accuracy: 0.9783
Epoch 100/100
844/844 - 3s - loss: 0.0558 - accuracy: 0.9828 - val_loss: 0.0749 - val_accuracy: 0.9778
evaluation start!
157/157 - 0s - loss: 0.0657 - accuracy: 0.9792

```


```python

```

### Optimizer


```python
adam_optimizer = tf.optimizers.Adam(learning_rate=1e-03)
```


```python
Deeper_CNN_model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Flatten(),
    keras.layers.Dense(256, activation = "relu"),
    keras.layers.Dense(128, activation = "relu"),
    keras.layers.Dense(10, activation = "softmax"),
])

Deeper_CNN_model.compile(loss = loss, optimizer = adam_optimizer, metrics = metrics)

with tf.device(device_name = device_name):
    print("training start!")
    hist = Deeper_CNN_model.fit(x_train, y_train, batch_size = BATCH_SIZE, epochs = EPOCHS, shuffle = True, verbose = 2, validation_data=(x_val, y_val))
    print("evaluation start!")
    Deeper_CNN_model.evaluate(x_test, y_test, batch_size = BATCH_SIZE, verbose = 2)

#show_model_output(x_test[0], Deeper_CNN_model)
```

```console
training start!
Epoch 1/100
844/844 - 3s - loss: 0.2361 - accuracy: 0.9287 - val_loss: 0.0772 - val_accuracy: 0.9747
Epoch 2/100
844/844 - 3s - loss: 0.0742 - accuracy: 0.9769 - val_loss: 0.0580 - val_accuracy: 0.9817
Epoch 3/100
844/844 - 3s - loss: 0.0528 - accuracy: 0.9837 - val_loss: 0.0484 - val_accuracy: 0.9850
Epoch 4/100
844/844 - 3s - loss: 0.0427 - accuracy: 0.9866 - val_loss: 0.0492 - val_accuracy: 0.9860
Epoch 5/100
844/844 - 3s - loss: 0.0353 - accuracy: 0.9889 - val_loss: 0.0480 - val_accuracy: 0.9848
Epoch 6/100
844/844 - 3s - loss: 0.0272 - accuracy: 0.9913 - val_loss: 0.0449 - val_accuracy: 0.9863
Epoch 7/100
844/844 - 3s - loss: 0.0239 - accuracy: 0.9927 - val_loss: 0.0566 - val_accuracy: 0.9838
Epoch 8/100
844/844 - 3s - loss: 0.0216 - accuracy: 0.9932 - val_loss: 0.0566 - val_accuracy: 0.9833
Epoch 9/100
844/844 - 3s - loss: 0.0188 - accuracy: 0.9939 - val_loss: 0.0514 - val_accuracy: 0.9877
Epoch 10/100
844/844 - 3s - loss: 0.0150 - accuracy: 0.9950 - val_loss: 0.0465 - val_accuracy: 0.9882
Epoch 11/100
844/844 - 3s - loss: 0.0140 - accuracy: 0.9956 - val_loss: 0.0908 - val_accuracy: 0.9780
Epoch 12/100
844/844 - 3s - loss: 0.0127 - accuracy: 0.9963 - val_loss: 0.0352 - val_accuracy: 0.9915
Epoch 13/100
844/844 - 3s - loss: 0.0119 - accuracy: 0.9963 - val_loss: 0.0776 - val_accuracy: 0.9823
Epoch 14/100
844/844 - 3s - loss: 0.0110 - accuracy: 0.9963 - val_loss: 0.0630 - val_accuracy: 0.9848
Epoch 15/100
844/844 - 3s - loss: 0.0103 - accuracy: 0.9967 - val_loss: 0.0642 - val_accuracy: 0.9882
Epoch 16/100
844/844 - 3s - loss: 0.0088 - accuracy: 0.9976 - val_loss: 0.0540 - val_accuracy: 0.9885
Epoch 17/100
844/844 - 3s - loss: 0.0084 - accuracy: 0.9974 - val_loss: 0.0585 - val_accuracy: 0.9862
Epoch 18/100
844/844 - 3s - loss: 0.0104 - accuracy: 0.9966 - val_loss: 0.0540 - val_accuracy: 0.9870
Epoch 19/100
844/844 - 3s - loss: 0.0067 - accuracy: 0.9977 - val_loss: 0.0654 - val_accuracy: 0.9855
Epoch 20/100
844/844 - 3s - loss: 0.0089 - accuracy: 0.9973 - val_loss: 0.0574 - val_accuracy: 0.9882
Epoch 21/100
844/844 - 3s - loss: 0.0060 - accuracy: 0.9982 - val_loss: 0.0591 - val_accuracy: 0.9868
Epoch 22/100
844/844 - 3s - loss: 0.0049 - accuracy: 0.9984 - val_loss: 0.0591 - val_accuracy: 0.9905
Epoch 23/100
844/844 - 3s - loss: 0.0085 - accuracy: 0.9974 - val_loss: 0.0669 - val_accuracy: 0.9867
Epoch 24/100
844/844 - 3s - loss: 0.0060 - accuracy: 0.9982 - val_loss: 0.0636 - val_accuracy: 0.9872
Epoch 25/100
844/844 - 3s - loss: 0.0053 - accuracy: 0.9982 - val_loss: 0.0651 - val_accuracy: 0.9873
Epoch 26/100
844/844 - 3s - loss: 0.0054 - accuracy: 0.9984 - val_loss: 0.0810 - val_accuracy: 0.9848
Epoch 27/100
844/844 - 3s - loss: 0.0069 - accuracy: 0.9980 - val_loss: 0.0621 - val_accuracy: 0.9878
Epoch 28/100
844/844 - 3s - loss: 0.0039 - accuracy: 0.9988 - val_loss: 0.0594 - val_accuracy: 0.9898
Epoch 29/100
844/844 - 3s - loss: 0.0068 - accuracy: 0.9978 - val_loss: 0.0655 - val_accuracy: 0.9872
Epoch 30/100
844/844 - 3s - loss: 0.0047 - accuracy: 0.9986 - val_loss: 0.0555 - val_accuracy: 0.9882
Epoch 31/100
844/844 - 3s - loss: 0.0034 - accuracy: 0.9991 - val_loss: 0.1105 - val_accuracy: 0.9827
Epoch 32/100
844/844 - 3s - loss: 0.0064 - accuracy: 0.9981 - val_loss: 0.0610 - val_accuracy: 0.9870
Epoch 33/100
844/844 - 3s - loss: 0.0042 - accuracy: 0.9989 - val_loss: 0.0616 - val_accuracy: 0.9887
Epoch 34/100
844/844 - 3s - loss: 0.0046 - accuracy: 0.9985 - val_loss: 0.0806 - val_accuracy: 0.9875
Epoch 35/100
844/844 - 3s - loss: 0.0060 - accuracy: 0.9985 - val_loss: 0.0776 - val_accuracy: 0.9865
Epoch 36/100
844/844 - 3s - loss: 0.0044 - accuracy: 0.9988 - val_loss: 0.0778 - val_accuracy: 0.9897
Epoch 37/100
844/844 - 3s - loss: 0.0041 - accuracy: 0.9988 - val_loss: 0.1107 - val_accuracy: 0.9832
Epoch 38/100
844/844 - 3s - loss: 0.0034 - accuracy: 0.9989 - val_loss: 0.0877 - val_accuracy: 0.9868
Epoch 39/100
844/844 - 3s - loss: 0.0062 - accuracy: 0.9983 - val_loss: 0.0844 - val_accuracy: 0.9862
Epoch 40/100
844/844 - 3s - loss: 0.0049 - accuracy: 0.9988 - val_loss: 0.1050 - val_accuracy: 0.9873
Epoch 41/100
844/844 - 3s - loss: 0.0042 - accuracy: 0.9986 - val_loss: 0.0964 - val_accuracy: 0.9867
Epoch 42/100
844/844 - 3s - loss: 0.0047 - accuracy: 0.9987 - val_loss: 0.0776 - val_accuracy: 0.9870
Epoch 43/100
844/844 - 3s - loss: 0.0030 - accuracy: 0.9991 - val_loss: 0.0843 - val_accuracy: 0.9863
Epoch 44/100
844/844 - 3s - loss: 0.0044 - accuracy: 0.9988 - val_loss: 0.0770 - val_accuracy: 0.9867
Epoch 45/100
844/844 - 3s - loss: 0.0045 - accuracy: 0.9987 - val_loss: 0.0769 - val_accuracy: 0.9877
Epoch 46/100
844/844 - 3s - loss: 0.0033 - accuracy: 0.9993 - val_loss: 0.0882 - val_accuracy: 0.9852
Epoch 47/100
844/844 - 3s - loss: 0.0035 - accuracy: 0.9990 - val_loss: 0.0937 - val_accuracy: 0.9867
Epoch 48/100
844/844 - 3s - loss: 0.0051 - accuracy: 0.9988 - val_loss: 0.0904 - val_accuracy: 0.9862
Epoch 49/100
844/844 - 3s - loss: 0.0039 - accuracy: 0.9989 - val_loss: 0.0759 - val_accuracy: 0.9893
Epoch 50/100
844/844 - 3s - loss: 0.0033 - accuracy: 0.9992 - val_loss: 0.0871 - val_accuracy: 0.9888
Epoch 51/100
844/844 - 3s - loss: 0.0033 - accuracy: 0.9991 - val_loss: 0.0738 - val_accuracy: 0.9872
Epoch 52/100
844/844 - 3s - loss: 0.0036 - accuracy: 0.9989 - val_loss: 0.0724 - val_accuracy: 0.9905
Epoch 53/100
844/844 - 3s - loss: 0.0042 - accuracy: 0.9991 - val_loss: 0.0831 - val_accuracy: 0.9892
Epoch 54/100
844/844 - 3s - loss: 0.0029 - accuracy: 0.9991 - val_loss: 0.0854 - val_accuracy: 0.9883
Epoch 55/100
844/844 - 3s - loss: 0.0038 - accuracy: 0.9992 - val_loss: 0.1101 - val_accuracy: 0.9853
Epoch 56/100
844/844 - 3s - loss: 0.0036 - accuracy: 0.9990 - val_loss: 0.1120 - val_accuracy: 0.9848
Epoch 57/100
844/844 - 3s - loss: 0.0050 - accuracy: 0.9987 - val_loss: 0.0723 - val_accuracy: 0.9897
Epoch 58/100
844/844 - 3s - loss: 0.0019 - accuracy: 0.9995 - val_loss: 0.0724 - val_accuracy: 0.9895
Epoch 59/100
844/844 - 3s - loss: 0.0036 - accuracy: 0.9990 - val_loss: 0.0915 - val_accuracy: 0.9885
Epoch 60/100
844/844 - 3s - loss: 0.0054 - accuracy: 0.9986 - val_loss: 0.0767 - val_accuracy: 0.9892
Epoch 61/100
844/844 - 3s - loss: 0.0026 - accuracy: 0.9993 - val_loss: 0.0896 - val_accuracy: 0.9868
Epoch 62/100
844/844 - 3s - loss: 0.0023 - accuracy: 0.9993 - val_loss: 0.0822 - val_accuracy: 0.9888
Epoch 63/100
844/844 - 3s - loss: 0.0017 - accuracy: 0.9995 - val_loss: 0.1177 - val_accuracy: 0.9877
Epoch 64/100
844/844 - 3s - loss: 0.0046 - accuracy: 0.9991 - val_loss: 0.1005 - val_accuracy: 0.9882
Epoch 65/100
844/844 - 3s - loss: 0.0040 - accuracy: 0.9991 - val_loss: 0.1046 - val_accuracy: 0.9877
Epoch 66/100
844/844 - 3s - loss: 0.0050 - accuracy: 0.9989 - val_loss: 0.0884 - val_accuracy: 0.9900
Epoch 67/100
844/844 - 3s - loss: 0.0029 - accuracy: 0.9992 - val_loss: 0.1097 - val_accuracy: 0.9892
Epoch 68/100
844/844 - 3s - loss: 0.0025 - accuracy: 0.9995 - val_loss: 0.1156 - val_accuracy: 0.9867
Epoch 69/100
844/844 - 3s - loss: 0.0023 - accuracy: 0.9994 - val_loss: 0.0909 - val_accuracy: 0.9908
Epoch 70/100
844/844 - 3s - loss: 0.0030 - accuracy: 0.9993 - val_loss: 0.0915 - val_accuracy: 0.9893
Epoch 71/100
844/844 - 3s - loss: 4.8473e-04 - accuracy: 0.9999 - val_loss: 0.1039 - val_accuracy: 0.9897
Epoch 72/100
844/844 - 3s - loss: 0.0055 - accuracy: 0.9987 - val_loss: 0.0885 - val_accuracy: 0.9873
Epoch 73/100
844/844 - 3s - loss: 0.0057 - accuracy: 0.9987 - val_loss: 0.0796 - val_accuracy: 0.9898
Epoch 74/100
844/844 - 3s - loss: 0.0011 - accuracy: 0.9997 - val_loss: 0.1501 - val_accuracy: 0.9828
Epoch 75/100
844/844 - 3s - loss: 0.0035 - accuracy: 0.9993 - val_loss: 0.1042 - val_accuracy: 0.9888
Epoch 76/100
844/844 - 3s - loss: 0.0018 - accuracy: 0.9996 - val_loss: 0.0966 - val_accuracy: 0.9882
Epoch 77/100
844/844 - 3s - loss: 0.0039 - accuracy: 0.9992 - val_loss: 0.0901 - val_accuracy: 0.9882
Epoch 78/100
844/844 - 3s - loss: 0.0032 - accuracy: 0.9993 - val_loss: 0.0946 - val_accuracy: 0.9892
Epoch 79/100
844/844 - 3s - loss: 0.0021 - accuracy: 0.9993 - val_loss: 0.1228 - val_accuracy: 0.9893
Epoch 80/100
844/844 - 3s - loss: 0.0056 - accuracy: 0.9989 - val_loss: 0.0954 - val_accuracy: 0.9888
Epoch 81/100
844/844 - 3s - loss: 0.0030 - accuracy: 0.9993 - val_loss: 0.1070 - val_accuracy: 0.9870
Epoch 82/100
844/844 - 3s - loss: 0.0012 - accuracy: 0.9997 - val_loss: 0.1214 - val_accuracy: 0.9882
Epoch 83/100
844/844 - 3s - loss: 0.0037 - accuracy: 0.9996 - val_loss: 0.0825 - val_accuracy: 0.9887
Epoch 84/100
844/844 - 3s - loss: 0.0035 - accuracy: 0.9991 - val_loss: 0.0963 - val_accuracy: 0.9902
Epoch 85/100
844/844 - 3s - loss: 0.0052 - accuracy: 0.9990 - val_loss: 0.0987 - val_accuracy: 0.9897
Epoch 86/100
844/844 - 3s - loss: 0.0026 - accuracy: 0.9994 - val_loss: 0.1146 - val_accuracy: 0.9882
Epoch 87/100
844/844 - 3s - loss: 0.0035 - accuracy: 0.9992 - val_loss: 0.1012 - val_accuracy: 0.9882
Epoch 88/100
844/844 - 3s - loss: 0.0014 - accuracy: 0.9995 - val_loss: 0.1139 - val_accuracy: 0.9902
Epoch 89/100
844/844 - 3s - loss: 0.0044 - accuracy: 0.9992 - val_loss: 0.1205 - val_accuracy: 0.9862
Epoch 90/100
844/844 - 3s - loss: 0.0018 - accuracy: 0.9995 - val_loss: 0.1263 - val_accuracy: 0.9878
Epoch 91/100
844/844 - 3s - loss: 2.5179e-04 - accuracy: 0.9999 - val_loss: 0.1254 - val_accuracy: 0.9897
Epoch 92/100
844/844 - 3s - loss: 6.9501e-06 - accuracy: 1.0000 - val_loss: 0.1301 - val_accuracy: 0.9898
Epoch 93/100
844/844 - 3s - loss: 1.4133e-06 - accuracy: 1.0000 - val_loss: 0.1327 - val_accuracy: 0.9897
Epoch 94/100
844/844 - 3s - loss: 7.0120e-07 - accuracy: 1.0000 - val_loss: 0.1348 - val_accuracy: 0.9902
Epoch 95/100
844/844 - 3s - loss: 4.4713e-07 - accuracy: 1.0000 - val_loss: 0.1371 - val_accuracy: 0.9903
Epoch 96/100
844/844 - 3s - loss: 2.9067e-07 - accuracy: 1.0000 - val_loss: 0.1398 - val_accuracy: 0.9903
Epoch 97/100
844/844 - 3s - loss: 1.8384e-07 - accuracy: 1.0000 - val_loss: 0.1429 - val_accuracy: 0.9903
Epoch 98/100
844/844 - 3s - loss: 1.1461e-07 - accuracy: 1.0000 - val_loss: 0.1461 - val_accuracy: 0.9903
Epoch 99/100
844/844 - 3s - loss: 7.0378e-08 - accuracy: 1.0000 - val_loss: 0.1494 - val_accuracy: 0.9902
Epoch 100/100
844/844 - 3s - loss: 4.3608e-08 - accuracy: 1.0000 - val_loss: 0.1524 - val_accuracy: 0.9902
evaluation start!
157/157 - 0s - loss: 0.1478 - accuracy: 0.9898

```


```python

```

### Batch Normalization Layer


```python
Deeper_CNN_model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), padding = "same", input_shape=(28, 28, 1)),
    keras.layers.Conv2D(32, (3, 3), padding = "same", input_shape=(28, 28, 1)),
    keras.layers.ReLU(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.BatchNormalization(),
    
    keras.layers.Conv2D(64, (3, 3), padding = "same"),
    keras.layers.Conv2D(64, (3, 3), padding = "same"),
    keras.layers.ReLU(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(128, (3, 3), padding = "same"),
    keras.layers.Conv2D(128, (3, 3), padding = "same"),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.BatchNormalization(),

    keras.layers.Flatten(),
    keras.layers.Dense(256),
    keras.layers.Dense(128),
    keras.layers.Dense(10),
    keras.layers.Softmax()
])

Deeper_CNN_model.summary()
Deeper_CNN_model.compile(loss = loss, optimizer = adam_optimizer, metrics = metrics)

with tf.device(device_name = device_name):
    print("training start!")
    hist = Deeper_CNN_model.fit(x_train, y_train, batch_size = BATCH_SIZE, epochs = EPOCHS, shuffle = True, verbose = 2, validation_data=(x_val, y_val))
    print("evaluation start!")
    Deeper_CNN_model.evaluate(x_test, y_test, batch_size = BATCH_SIZE, verbose = 2)

#show_model_output(x_test[0], Deeper_CNN_model)
```

```console
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_3 (Conv2D)            (None, 28, 28, 32)        320       
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 28, 28, 32)        9248      
_________________________________________________________________
re_lu_2 (ReLU)               (None, 28, 28, 32)        0         
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 14, 14, 32)        0         
_________________________________________________________________
batch_normalization_3 (Batch (None, 14, 14, 32)        128       
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 14, 14, 64)        18496     
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 14, 14, 64)        36928     
_________________________________________________________________
re_lu_3 (ReLU)               (None, 14, 14, 64)        0         
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 7, 7, 64)          0         
_________________________________________________________________
batch_normalization_4 (Batch (None, 7, 7, 64)          256       
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 7, 7, 128)         73856     
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 7, 7, 128)         147584    
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 3, 3, 128)         0         
_________________________________________________________________
batch_normalization_5 (Batch (None, 3, 3, 128)         512       
_________________________________________________________________
flatten_1 (Flatten)          (None, 1152)              0         
_________________________________________________________________
dense_3 (Dense)              (None, 256)               295168    
_________________________________________________________________
dense_4 (Dense)              (None, 128)               32896     
_________________________________________________________________
dense_5 (Dense)              (None, 10)                1290      
_________________________________________________________________
softmax_1 (Softmax)          (None, 10)                0         
=================================================================
Total params: 616,682
Trainable params: 616,234
Non-trainable params: 448
_________________________________________________________________
training start!
Epoch 1/100
844/844 - 6s - loss: 0.3811 - accuracy: 0.9464 - val_loss: 0.0879 - val_accuracy: 0.9748
Epoch 2/100
844/844 - 5s - loss: 0.0670 - accuracy: 0.9801 - val_loss: 0.1066 - val_accuracy: 0.9722
Epoch 3/100
844/844 - 5s - loss: 0.0583 - accuracy: 0.9825 - val_loss: 0.0550 - val_accuracy: 0.9855
Epoch 4/100
844/844 - 5s - loss: 0.0533 - accuracy: 0.9844 - val_loss: 0.0783 - val_accuracy: 0.9783
Epoch 5/100
844/844 - 5s - loss: 0.0492 - accuracy: 0.9857 - val_loss: 0.0993 - val_accuracy: 0.9753
Epoch 6/100
844/844 - 5s - loss: 0.0443 - accuracy: 0.9872 - val_loss: 0.0489 - val_accuracy: 0.9842
Epoch 7/100
844/844 - 5s - loss: 0.0388 - accuracy: 0.9883 - val_loss: 0.0791 - val_accuracy: 0.9812
Epoch 8/100
844/844 - 5s - loss: 0.0344 - accuracy: 0.9895 - val_loss: 0.0492 - val_accuracy: 0.9892
Epoch 9/100
844/844 - 5s - loss: 0.0320 - accuracy: 0.9902 - val_loss: 0.0465 - val_accuracy: 0.9873
Epoch 10/100
844/844 - 5s - loss: 0.0289 - accuracy: 0.9905 - val_loss: 0.0618 - val_accuracy: 0.9823
Epoch 11/100
844/844 - 5s - loss: 0.0242 - accuracy: 0.9930 - val_loss: 0.0427 - val_accuracy: 0.9892
Epoch 12/100
844/844 - 5s - loss: 0.0221 - accuracy: 0.9933 - val_loss: 0.0478 - val_accuracy: 0.9872
Epoch 13/100
844/844 - 5s - loss: 0.0230 - accuracy: 0.9928 - val_loss: 0.0603 - val_accuracy: 0.9833
Epoch 14/100
844/844 - 5s - loss: 0.0171 - accuracy: 0.9949 - val_loss: 0.0320 - val_accuracy: 0.9928
Epoch 15/100
844/844 - 5s - loss: 0.0218 - accuracy: 0.9932 - val_loss: 0.0412 - val_accuracy: 0.9910
Epoch 16/100
844/844 - 5s - loss: 0.0153 - accuracy: 0.9951 - val_loss: 0.0471 - val_accuracy: 0.9897
Epoch 17/100
844/844 - 5s - loss: 0.0165 - accuracy: 0.9950 - val_loss: 0.0332 - val_accuracy: 0.9913
Epoch 18/100
844/844 - 5s - loss: 0.0141 - accuracy: 0.9954 - val_loss: 0.0671 - val_accuracy: 0.9863
Epoch 19/100
844/844 - 5s - loss: 0.0159 - accuracy: 0.9952 - val_loss: 0.0461 - val_accuracy: 0.9908
Epoch 20/100
844/844 - 5s - loss: 0.0084 - accuracy: 0.9975 - val_loss: 0.0645 - val_accuracy: 0.9857
Epoch 21/100
844/844 - 5s - loss: 0.0148 - accuracy: 0.9956 - val_loss: 0.0517 - val_accuracy: 0.9907
Epoch 22/100
844/844 - 5s - loss: 0.0117 - accuracy: 0.9967 - val_loss: 0.0453 - val_accuracy: 0.9907
Epoch 23/100
844/844 - 5s - loss: 0.0063 - accuracy: 0.9980 - val_loss: 0.0464 - val_accuracy: 0.9918
Epoch 24/100
844/844 - 5s - loss: 0.0159 - accuracy: 0.9952 - val_loss: 0.0543 - val_accuracy: 0.9915
Epoch 25/100
844/844 - 5s - loss: 0.0101 - accuracy: 0.9971 - val_loss: 0.0457 - val_accuracy: 0.9907
Epoch 26/100
844/844 - 5s - loss: 0.0059 - accuracy: 0.9980 - val_loss: 0.0458 - val_accuracy: 0.9905
Epoch 27/100
844/844 - 5s - loss: 0.0104 - accuracy: 0.9970 - val_loss: 0.1965 - val_accuracy: 0.9653
Epoch 28/100
844/844 - 5s - loss: 0.0115 - accuracy: 0.9964 - val_loss: 0.0795 - val_accuracy: 0.9845
Epoch 29/100
844/844 - 5s - loss: 0.0069 - accuracy: 0.9981 - val_loss: 0.0513 - val_accuracy: 0.9908
Epoch 30/100
844/844 - 5s - loss: 0.0086 - accuracy: 0.9974 - val_loss: 0.0573 - val_accuracy: 0.9900
Epoch 31/100
844/844 - 5s - loss: 0.0056 - accuracy: 0.9981 - val_loss: 0.0495 - val_accuracy: 0.9918
Epoch 32/100
844/844 - 5s - loss: 0.0088 - accuracy: 0.9975 - val_loss: 0.0726 - val_accuracy: 0.9892
Epoch 33/100
844/844 - 5s - loss: 0.0052 - accuracy: 0.9984 - val_loss: 0.0505 - val_accuracy: 0.9890
Epoch 34/100
844/844 - 5s - loss: 0.0068 - accuracy: 0.9980 - val_loss: 0.0598 - val_accuracy: 0.9913
Epoch 35/100
844/844 - 5s - loss: 0.0091 - accuracy: 0.9973 - val_loss: 0.0459 - val_accuracy: 0.9923
Epoch 36/100
844/844 - 5s - loss: 0.0041 - accuracy: 0.9986 - val_loss: 0.0517 - val_accuracy: 0.9923
Epoch 37/100
844/844 - 5s - loss: 0.0063 - accuracy: 0.9982 - val_loss: 0.0605 - val_accuracy: 0.9910
Epoch 38/100
844/844 - 5s - loss: 0.0065 - accuracy: 0.9980 - val_loss: 0.0649 - val_accuracy: 0.9880
Epoch 39/100
844/844 - 5s - loss: 0.0070 - accuracy: 0.9978 - val_loss: 0.0480 - val_accuracy: 0.9928
Epoch 40/100
844/844 - 5s - loss: 0.0037 - accuracy: 0.9989 - val_loss: 0.0629 - val_accuracy: 0.9898
Epoch 41/100
844/844 - 5s - loss: 0.0090 - accuracy: 0.9975 - val_loss: 0.0734 - val_accuracy: 0.9898
Epoch 42/100
844/844 - 5s - loss: 0.0049 - accuracy: 0.9986 - val_loss: 0.0497 - val_accuracy: 0.9903
Epoch 43/100
844/844 - 5s - loss: 0.0028 - accuracy: 0.9991 - val_loss: 0.0524 - val_accuracy: 0.9910
Epoch 44/100
844/844 - 5s - loss: 0.0036 - accuracy: 0.9989 - val_loss: 0.0600 - val_accuracy: 0.9898
Epoch 45/100
844/844 - 5s - loss: 0.0085 - accuracy: 0.9978 - val_loss: 0.0454 - val_accuracy: 0.9925
Epoch 46/100
844/844 - 5s - loss: 0.0040 - accuracy: 0.9988 - val_loss: 0.0801 - val_accuracy: 0.9863
Epoch 47/100
844/844 - 5s - loss: 0.0036 - accuracy: 0.9988 - val_loss: 0.0548 - val_accuracy: 0.9927
Epoch 48/100
844/844 - 5s - loss: 0.0063 - accuracy: 0.9981 - val_loss: 0.0508 - val_accuracy: 0.9923
Epoch 49/100
844/844 - 5s - loss: 0.0046 - accuracy: 0.9986 - val_loss: 0.0616 - val_accuracy: 0.9908
Epoch 50/100
844/844 - 5s - loss: 0.0040 - accuracy: 0.9989 - val_loss: 0.0499 - val_accuracy: 0.9913
Epoch 51/100
844/844 - 5s - loss: 0.0048 - accuracy: 0.9986 - val_loss: 0.0608 - val_accuracy: 0.9915
Epoch 52/100
844/844 - 5s - loss: 0.0030 - accuracy: 0.9991 - val_loss: 0.0618 - val_accuracy: 0.9918
Epoch 53/100
844/844 - 5s - loss: 0.0069 - accuracy: 0.9982 - val_loss: 0.0533 - val_accuracy: 0.9915
Epoch 54/100
844/844 - 5s - loss: 0.0021 - accuracy: 0.9991 - val_loss: 0.0608 - val_accuracy: 0.9918
Epoch 55/100
844/844 - 5s - loss: 0.0055 - accuracy: 0.9983 - val_loss: 0.0489 - val_accuracy: 0.9915
Epoch 56/100
844/844 - 5s - loss: 0.0025 - accuracy: 0.9992 - val_loss: 0.0704 - val_accuracy: 0.9888
Epoch 57/100
844/844 - 5s - loss: 0.0044 - accuracy: 0.9988 - val_loss: 0.0740 - val_accuracy: 0.9903
Epoch 58/100
844/844 - 5s - loss: 0.0039 - accuracy: 0.9990 - val_loss: 0.1171 - val_accuracy: 0.9863
Epoch 59/100
844/844 - 5s - loss: 0.0058 - accuracy: 0.9984 - val_loss: 0.0563 - val_accuracy: 0.9918
Epoch 60/100
844/844 - 5s - loss: 0.0024 - accuracy: 0.9992 - val_loss: 0.0505 - val_accuracy: 0.9930
Epoch 61/100
844/844 - 5s - loss: 0.0037 - accuracy: 0.9990 - val_loss: 0.0484 - val_accuracy: 0.9917
Epoch 62/100
844/844 - 5s - loss: 0.0038 - accuracy: 0.9986 - val_loss: 0.0464 - val_accuracy: 0.9925
Epoch 63/100
844/844 - 5s - loss: 0.0032 - accuracy: 0.9991 - val_loss: 0.0534 - val_accuracy: 0.9932
Epoch 64/100
844/844 - 5s - loss: 0.0056 - accuracy: 0.9986 - val_loss: 0.0529 - val_accuracy: 0.9930
Epoch 65/100
844/844 - 5s - loss: 0.0030 - accuracy: 0.9992 - val_loss: 0.0599 - val_accuracy: 0.9938
Epoch 66/100
844/844 - 5s - loss: 0.0015 - accuracy: 0.9995 - val_loss: 0.0815 - val_accuracy: 0.9893
Epoch 67/100
844/844 - 5s - loss: 0.0049 - accuracy: 0.9987 - val_loss: 0.0731 - val_accuracy: 0.9915
Epoch 68/100
844/844 - 5s - loss: 0.0051 - accuracy: 0.9987 - val_loss: 0.0710 - val_accuracy: 0.9922
Epoch 69/100
844/844 - 5s - loss: 0.0028 - accuracy: 0.9992 - val_loss: 0.0561 - val_accuracy: 0.9922
Epoch 70/100
844/844 - 5s - loss: 0.0045 - accuracy: 0.9988 - val_loss: 0.0575 - val_accuracy: 0.9920
Epoch 71/100
844/844 - 5s - loss: 0.0026 - accuracy: 0.9994 - val_loss: 0.0510 - val_accuracy: 0.9922
Epoch 72/100
844/844 - 5s - loss: 0.0020 - accuracy: 0.9994 - val_loss: 0.0638 - val_accuracy: 0.9915
Epoch 73/100
844/844 - 5s - loss: 0.0066 - accuracy: 0.9985 - val_loss: 0.0735 - val_accuracy: 0.9893
Epoch 74/100
844/844 - 5s - loss: 0.0029 - accuracy: 0.9992 - val_loss: 0.0595 - val_accuracy: 0.9933
Epoch 75/100
844/844 - 5s - loss: 1.8501e-04 - accuracy: 1.0000 - val_loss: 0.0492 - val_accuracy: 0.9933
Epoch 76/100
844/844 - 5s - loss: 6.9908e-05 - accuracy: 1.0000 - val_loss: 0.0548 - val_accuracy: 0.9925
Epoch 77/100
844/844 - 5s - loss: 5.6622e-05 - accuracy: 1.0000 - val_loss: 0.0563 - val_accuracy: 0.9938
Epoch 78/100
844/844 - 5s - loss: 5.2487e-06 - accuracy: 1.0000 - val_loss: 0.0561 - val_accuracy: 0.9942
Epoch 79/100
844/844 - 5s - loss: 8.4469e-04 - accuracy: 0.9998 - val_loss: 0.1403 - val_accuracy: 0.9832
Epoch 80/100
844/844 - 5s - loss: 0.0133 - accuracy: 0.9968 - val_loss: 0.0553 - val_accuracy: 0.9937
Epoch 81/100
844/844 - 5s - loss: 0.0022 - accuracy: 0.9995 - val_loss: 0.0530 - val_accuracy: 0.9932
Epoch 82/100
844/844 - 5s - loss: 3.1437e-04 - accuracy: 0.9999 - val_loss: 0.0520 - val_accuracy: 0.9933
Epoch 83/100
844/844 - 5s - loss: 0.0046 - accuracy: 0.9989 - val_loss: 0.0724 - val_accuracy: 0.9912
Epoch 84/100
844/844 - 5s - loss: 0.0037 - accuracy: 0.9990 - val_loss: 0.0609 - val_accuracy: 0.9927
Epoch 85/100
844/844 - 5s - loss: 0.0027 - accuracy: 0.9993 - val_loss: 0.0504 - val_accuracy: 0.9923
Epoch 86/100
844/844 - 5s - loss: 0.0030 - accuracy: 0.9991 - val_loss: 0.0549 - val_accuracy: 0.9937
Epoch 87/100
844/844 - 5s - loss: 0.0043 - accuracy: 0.9989 - val_loss: 0.0947 - val_accuracy: 0.9898
Epoch 88/100
844/844 - 5s - loss: 0.0036 - accuracy: 0.9991 - val_loss: 0.0741 - val_accuracy: 0.9912
Epoch 89/100
844/844 - 5s - loss: 0.0013 - accuracy: 0.9995 - val_loss: 0.0624 - val_accuracy: 0.9920
Epoch 90/100
844/844 - 5s - loss: 0.0029 - accuracy: 0.9993 - val_loss: 0.0677 - val_accuracy: 0.9927
Epoch 91/100
844/844 - 5s - loss: 0.0043 - accuracy: 0.9989 - val_loss: 0.0717 - val_accuracy: 0.9913
Epoch 92/100
844/844 - 5s - loss: 0.0031 - accuracy: 0.9993 - val_loss: 0.0833 - val_accuracy: 0.9915
Epoch 93/100
844/844 - 5s - loss: 0.0011 - accuracy: 0.9997 - val_loss: 0.0610 - val_accuracy: 0.9930
Epoch 94/100
844/844 - 5s - loss: 0.0029 - accuracy: 0.9992 - val_loss: 0.0990 - val_accuracy: 0.9892
Epoch 95/100
844/844 - 5s - loss: 0.0038 - accuracy: 0.9990 - val_loss: 0.0654 - val_accuracy: 0.9920
Epoch 96/100
844/844 - 5s - loss: 0.0020 - accuracy: 0.9996 - val_loss: 0.0614 - val_accuracy: 0.9928
Epoch 97/100
844/844 - 5s - loss: 0.0031 - accuracy: 0.9992 - val_loss: 0.0828 - val_accuracy: 0.9900
Epoch 98/100
844/844 - 5s - loss: 0.0033 - accuracy: 0.9993 - val_loss: 0.0772 - val_accuracy: 0.9907
Epoch 99/100
844/844 - 5s - loss: 0.0022 - accuracy: 0.9994 - val_loss: 0.0863 - val_accuracy: 0.9900
Epoch 100/100
844/844 - 5s - loss: 0.0043 - accuracy: 0.9989 - val_loss: 0.0691 - val_accuracy: 0.9928
evaluation start!
157/157 - 1s - loss: 0.0581 - accuracy: 0.9929

```

