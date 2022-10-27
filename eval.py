import argparse
import numpy as np 
import matplotlib.pyplot as plt 

import torch 
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchsummary import summary

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

def diplay_result(x, y, y_pred):
    plt.imshow(torch.permute(x[0], (1, 2, 0)), cmap='gray')
    plt.axis('off')
    plt.title(f"Actual: {y.item()}, Predicted: {y_pred.item()}")
    plt.show()

def eval(opt, model, test_loader):
    criterion = CrossEntropyLoss()
    xs = []
    ys = []
    y_preds = []
    test_correct, test_loss, test_total = 0, 0, 0
    cnt = 0
    with torch.no_grad():
        for batch in test_loader:
            cnt += 1 
            x, y = batch
            x = x.to(opt.device)
            y = y.to(opt.device)

            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.item()

            test_total += 1 
            test_correct += int(torch.argmax(y_hat.data, dim=1) == y)

            xs.append(x)
            ys.append(y)
            y_preds.append(torch.argmax(y_hat.data, dim=1))
            if cnt == opt.n_samples:
                break

    print(f'Test Loss: {test_loss}, Accuracy: {test_correct/test_total*100:.2f}%')
    return xs, ys, y_preds


if __name__ == '__main__':
    # Get arguments
    opt = get_args()

    # Load dataset into DataLoader
    test_load = []

    # Load model
    model = MyModel()

    model = load_model(model, opt.model_path)

    summary(model, (1, opt.input_size, opt.input_size))

    # Eval model
    xs, ys, y_preds = eval(opt, model, test_load)

    # Display results
    if opt.display:
        for x, y, y_pred in zip(xs, ys, y_preds):
            diplay_result(x, y, y_pred)

