import torch
import onnx
from train import CSRNetLightning
from config import Config

cfg = Config()
model_path = "checkpoints/epoch=156-val_mae=30.95.ckpt"
model = CSRNetLightning.load_from_checkpoint(checkpoint_path=model_path, config=cfg)

dummy_input = torch.randn(1, 3, 224, 224)  # Adjust the size as per your model
dummy_input = dummy_input.to(model.device) # Move dummy_input to device where model is
model.to(model.device) # Move model to same device where dummy_input is
torch.onnx.export(model, dummy_input, "model.onnx")
#

