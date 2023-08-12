import torch
import time
import numpy as np
from networks.resnet_big import *
from PIL import Image
from torch.quantization import get_default_qconfig
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torchvision import transforms

q_config = get_default_qconfig()
qconfig_dict = {"": q_config}

class EmoSameQT:
    def __init__(self, modelPath, QmodelPath):
        # try:
        float_model = torch.load(modelPath, map_location = "cpu")
        float_model.eval()
        p_model = prepare_fx(float_model, qconfig_dict, example_inputs = torch.randn(1, 3, 224, 224))
        self.model = convert_fx(p_model)
        self.model.load_state_dict(torch.load(QmodelPath, map_location = "cpu"))
        self.model.eval()
        # print(self.model)

        self.normalize = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.25, 0.25, 0.25))
        ])
        # except Exception as e:
        #     print("Failed to load model: {}".format(str(e)))

    def quantify(self, img_path: str) -> np.ndarray:
        try:
            im = self.normalize(Image.open(img_path).convert("RGB")).unsqueeze(0)
            with torch.no_grad(): 
                return np.array(self.model(im))
        except Exception as e:
            print("Failed to quantify: {}".format(str(e)))
            return np.array([])

    def quantify_in_batches(self, img_paths: list) -> np.ndarray:
        try:
            im = torch.stack([self.normalize(Image.open(img_path).convert("RGB")) for img_path in img_paths], dim = 0)
            with torch.no_grad(): 
                return np.array(self.model(im))
        except Exception as e:
            print("Failed to quantify: {}".format(str(e)))
            return np.array([])

    def cosine_similarity(self, vec1, vec2) -> float:
        if isinstance(vec1, torch.Tensor):
            return float(torch.cosine_similarity(vec1, vec2)[0])
        elif isinstance(vec1, np.ndarray):
            return float(torch.cosine_similarity(torch.tensor(vec1), torch.tensor(vec2))[0])
        else:
            return 0

QmodelPath = "D:/AdminFiles/Download/ckpt_epoch_100_rein.pth.qt (1).pth"
modelPath = "D:/AdminFiles/Download/ckpt_epoch_100_rein.pth"
img1p = "D:/test2/8.jpg"
img2p = "D:/test2/9.jpg"
img3p = "D:/test2/1.jpg"

emo = EmoSameQT(modelPath, QmodelPath)

start = time.time()
im1 = emo.quantify(img1p)
im2 = emo.quantify(img2p)
im3 = emo.quantify(img3p)
end = time.time()

print(end - start)
print(len(im1[0]))
print(emo.cosine_similarity(im1, im2))
print(emo.cosine_similarity(im1, im3))
print(emo.cosine_similarity(im2, im3))
