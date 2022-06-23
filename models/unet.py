import segmentation_models_pytorch as smp
import torch
from utils.config import CFG
from monai.networks.nets import UNet

def build_model(cfg):
    model = UNet(
    spatial_dims=3,
    in_channels=cfg.in_channels, #RGB or Grayscale
    out_channels=cfg.num_classes
    channels=(32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2),
    kernel_size=3,
    up_kernel_size=3,
    num_res_units=2,
    act="PRELU",
    norm="BATCH",
    dropout=0.2,
    bias=True,
    dimensions=None,
).to(cfg.device)

def load_model(path):
    model = build_model()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model