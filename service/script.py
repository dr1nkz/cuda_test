import torch
import onnxruntime
from dotenv import load_dotenv

print(onnxruntime.get_available_providers())
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
path = '/service/forklift_cabledrum_model/forklift_cabledrum_model.onnx.onnx'
session = onnxruntime.InferenceSession(path, providers=providers)
