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
from dataset import SimpleCrop
from torchvision import transforms
from PIL import Image


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

def setup():
    config = Config()
    model = CSRNetLightning.load_from_checkpoint('checkpoints/epoch=156-val_mae=30.95.ckpt', config=config, lr=1e-4, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    return model
def infer(input):
    model = setup()
    with torch.no_grad():
        input = input.astype(np.float32)
        pil_image = Image.fromarray(input) 
        plt.imsave('input.png', pil_image)
        pil_image = transforms.functional.crop(pil_image, 0, 0, 40, 0)
        input = transform(pil_image)
        input = np.array(input)
        print(input.shape)
        output = model(input)
        output = output.squeeze(0).squeeze(0).cpu().numpy()
        print(output.shape)
        count = np.sum(output)
        if os.path.exists('output.png'):
            os.remove('output.png')
        else:
            plt.imsave('output.png', output, cmap=CM.jet)
        return count
    


demo = gr.Interface(fn=infer, inputs=[
    gr.Image(source="webcam", type="numpy", label="Webcam", streaming=True),
], outputs="number", title="Crowd Counting", description="Count the number of people in an image using a deep learning model.")

demo.launch()