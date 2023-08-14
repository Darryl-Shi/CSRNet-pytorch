import torch
import numpy as np

preds = torch.load('output/predictions.pt')

count = np.sum(preds)
print(round(count))