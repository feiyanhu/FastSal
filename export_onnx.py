import model.fastSal as fastsal
from utils import load_weight
import torch
import time
import torch.onnx
import onnxruntime
from PIL import Image
import numpy as np

def exportToONNX(weights, model_type, output):

    # Fastsal - Type Coco A
    model = fastsal.fastsal(pretrain_mode=False, model_type=model_type)
    state_dict, opt_state = load_weight(weights, remove_decoder=False)
    model.load_state_dict(state_dict)
    model.eval()
    torch_out = model(x)
    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        output,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=11,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
    )


# Input weights
x = torch.zeros((1, 3, 192, 256))

print("Exporting FastSal Coco A to .onnx...")
exportToONNX("weights/coco_A.pth", "A", "onnx/cocoA/fastsal.onnx")

print("Exporting FastSal Coco C to .onnx...")
exportToONNX("weights/coco_C.pth", "C", "onnx/cocoC/fastsal.onnx")

print("Exporting FastSal Salicon A to .onnx...")
exportToONNX("weights/salicon_A.pth", "A", "onnx/saliconA/fastsal.onnx")

print("Exporting FastSal Salicon C to .onnx...")
exportToONNX("weights/salicon_C.pth", "C", "onnx/saliconC/fastsal.onnx")

print("Finished.")




