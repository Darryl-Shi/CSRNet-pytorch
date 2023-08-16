import gradio as gr
import torch
import pytorch_lightning as pl
from main import CSRNetLightning
from config import Config
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as CM
from cv2 import VideoCapture
import os
import time
from dataset import PairedCrop
from torchvision import transforms

transforms = transforms.Compose([
    PairedCrop(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])
    
def setup():
    config = Config()
    model = CSRNetLightning.load_from_checkpoint('checkpoints/epoch=156-val_mae=30.95.ckpt', config=config, lr=1e-4, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    return model
def infer(model, input):
    try:
        input = torch.from_numpy(input).unsqueeze(0)
        output = model(input)
        output = output.squeeze(0).squeeze(0).cpu().numpy()
        count = np.sum(output)
        if os.path.exists('output.png'):
            os.remove('output.png')
        else:
            plt.imsave('output.png', output, cmap=CM.jet)
        return count
        
    except Exception as e:
        print(e)
        return 0

with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("# Crowd Counting")
    with gr.Row():
        with gr.Accordion("Settings"):
            fpm = gr.Slider(minimum=1, maximum=60, step=1, value=15, label="Frames to process per minute", interactive=True)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("# Live webcam feed")
        with gr.Column():
            gr.Markdown("# Density map")
    with gr.Row():
        with gr.Column():
            webcam_feed = gr.Image(source="webcam", streaming=True)

        with gr.Column():
            cam_input = gr.Image(source="webcam", streaming=True, visible=False, width=224, height=224)
            output = gr.Image()

if __name__ == "__main__":
    model = setup()
    demo.launch(share=True)