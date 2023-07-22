import torch
import onnx
from train import CSRNetLightning
from config import Config

cfg = Config()
model = CSRNetLightning(cfg).load_from_checkpoint("checkpoints/epoch=156-val_mae=30.95.ckpt")

torch.onnx.export(model, "model.onnx")

#

