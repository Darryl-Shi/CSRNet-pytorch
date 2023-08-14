import torch
import matplotlib.pyplot as plt
import matplotlib.cm as CM
from tqdm import tqdm
import pytorch_lightning
import dataset
from dataset import CrowdDataset
from config import Config

from main import CSRNetLightning


def cal_mae(root,model_param_path):
    '''
    Calculate the MAE of the test data.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    '''
    cfg = Config()
    model = CSRNetLightning.load_from_checkpoint(checkpoint_path=model_param_path, config=cfg, lr=1e-4)
    dataloader= dataset.create_test_dataloader(root)
    model.eval()
    mae=0
    with torch.no_grad():
        for i,(img,gt_dmap) in enumerate(tqdm(dataloader)):
            # forward propagation
            et_dmap=model(img)
            mae+=abs(et_dmap.data.sum()-gt_dmap.data.sum()).item()
            del img,gt_dmap,et_dmap

    print("model_param_path:"+model_param_path+" mae:"+str(mae/len(dataloader)))

def estimate_density_map(root,model_param_path,index):
    '''
    Show one estimated density-map.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    index: the order of the test image in test dataset.
    '''
    cfg = Config()
    model = CSRNetLightning.load_from_checkpoint(checkpoint_path=model_param_path, config=cfg, lr=1e-4)
    dataloader= dataset.create_test_dataloader(root)
    model.eval()
    for i,(img,gt_dmap) in enumerate(dataloader):
        if i==index:
            # forward propagation
            et_dmap=model(img).detach()
            et_dmap=et_dmap.squeeze(0).squeeze(0).cpu().numpy()
            print(et_dmap.shape)
            plt.imshow(et_dmap,cmap=CM.jet)
            break


if __name__=="__main__":
    torch.backends.cudnn.enabled=False
    root='data/part_A_final'
    model_param_path='checkpoints/epoch=156-val_mae=30.95.ckpt'
    cal_mae(root,model_param_path)
    estimate_density_map(root,model_param_path,3) 