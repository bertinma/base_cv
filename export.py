import torch
from model.model import MyModel
import argparse
from pathlib import Path 
from torchsummary import summary

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="weights/model.pt")
    parser.add_argument('--format', type=str, default='onnx', choices=['onnx'])
    parser.add_argument("--input-size", type=int, default=224)
    return parser.parse_args()

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model


if __name__ == '__main__':
    opt = get_args()
    # Instantiate your model. This is just a regular PyTorch model that will be exported in the following steps.
    # Load model
    model = MyModel()
    # Evaluate the model to switch some operations from training mode to inference.
    model.eval()

    summary(model, (1, opt.input_size, opt.input_size))

    # Create dummy input for the model. It will be used to run the model inside export function.
    dummy_input = torch.randn(1, 1, opt.input_size, opt.input_size)


    model_name = Path(opt.model_path).stem
    # Call the export function
    if opt.format == 'onnx':
        torch.onnx.export(model, (dummy_input, ), f'weights/onnx/{model_name}.onnx')

    print('Export complete!')