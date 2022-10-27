import argparse
import random 
import torch
from torchsummary import summary
from icecream import ic 

# import model class 
from model.model import MyModel

torch.autograd.set_detect_anomaly(True)



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="weights/model.pt")
    parser.add_argument("--input-size", type=int, default=224)
    return parser.parse_args()

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model


def display_result(y_pred):
    pass


def detect(opt, model, test_loader):
    with torch.no_grad():
        cnt = 0 
        while cnt < opt.n_samples:
            batch = random.choice(test_loader.dataset)
            x, y = batch
            x = torch.unsqueeze(x, 0).to(opt.device)
            y = torch.Tensor([y]).to(opt.device)
            y_hat = model(x)

            # display result 
            display_result(y_hat)

if __name__ == '__main__':
    # Get arguments
    opt = get_args()

    # Load MNIST dataset into DataLoader
    test_load = []
    # Load model
    model = MyModel()

    model = load_model(model, opt.model_path)

    summary(model, (1, opt.input_size, opt.input_size))

    # Eval model
    detect(opt, model, test_load)

