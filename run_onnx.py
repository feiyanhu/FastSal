import torch
import time
import onnxruntime
import numpy as np

model_path = "onnx/cocoA/fastsal.onnx"
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# Initiate a new session
ort_session = onnxruntime.InferenceSession(model_path)

# Generate input of appropriate size
x = torch.zeros((1, 3, 192, 256))

startTime = time.time()
for i in range(0,100):
    t1 = time.time()
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    y = ort_outs[0]
    t2 = time.time()
    interval = (t2 - t1)
    fps = 1 / interval
    print("Interval : " + str(interval))
    print("FPS : " + str(fps))

average_interval = (time.time()-startTime)/100
average_FPS = 1/average_interval
print("Finished.")
print("Average interval : "+str(average_interval))
print("Average FPS : "+str(average_FPS))