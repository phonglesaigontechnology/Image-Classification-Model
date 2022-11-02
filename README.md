# Build a Deep Learning model for the Image Classification task in a simple way

Dependencies 
+ OS: Mac/Linux/Window
+ Python3 (version >= 3.7) [Download](https://www.python.org/downloads/)
+ Conda environment (latest version) [Download](https://conda.io/projects/conda/en/latest/user-guide/install/download.html)
+ Docker [Download](https://docs.docker.com/desktop/)

## 1. Setup 

Conda environment 

```bash
$ conda create -n my-env python=3.7
$ conda activate my-env
(my-env)$ pip install -r requirements.txt
```

## 2. Run script  

Train CNN model 

```bash 
(my-env)$ python train_cnn_model.py
```

Train Petrained model (VGG16)

```bash 
(my-env)$ python train_pretrained_model.py
```
