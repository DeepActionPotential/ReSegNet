import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
import cv2
from segmentation_models_pytorch import Unet


def preprocess_image(image, size=(512, 512)):
    image = image.resize(size)
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))
    tensor = torch.tensor(img_array).unsqueeze(0)
    return tensor

def predict_mask(model, tensor):
    with torch.no_grad():
        output = torch.sigmoid(model(tensor))
    return output.squeeze().cpu().numpy()

def postprocess_mask(mask_array, threshold=0.5):
    mask = (mask_array > threshold).astype(np.uint8) * 255
    mask_rgb = np.stack([mask]*3, axis=-1)
    return Image.fromarray(mask_rgb)


def load_model(path: str, device: torch.device):
    model = Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)  # your architecture
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
