# Title 

This repository implements... 
This repo uses PyTorch framework. 

## Model Architecture 

## Data 


## Training
To train the model
### Localy 
Install following dependencies : 
- torch==1.12.1
- torchsummary==1.5.1
- torchvision==0.13.1
- numpy==1.21.5
- matplotlib==3.5.1
- tqdm==4.64.1


### Using Docker 

- Run the following command in bash shell opened by docker : 
```bash
python train.py .... 
```

## Inference using ONNX format 
To convert the model to ONNX format, run the following command : 
```bash
python export.py 
```

