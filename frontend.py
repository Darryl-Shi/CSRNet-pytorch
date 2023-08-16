import gradio as gr
import torch
import pytorch_lightning as pl
from main import CSRNetLightning
from config import Config
import numpy as np

def setup():
    config = Config()
    model = CSRNetLightning.load_from_checkpoint('checkpoints/epoch=156-val_mae=30.95.ckpt', config=config, lr=1e-4)
    return model
def infer(input):
    model = setup()
    input = torch.from_numpy(input).unsqueeze(0)
    output = model(input)
    print(output.shape)


infer(np.random.rand(1, 3, 224, 224))