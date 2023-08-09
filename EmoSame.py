import torch
import time
import numpy as np
from networks.resnet_big import *
from PIL import Image
from torchvision import transforms

### model end

class EmoSame:
    def __init__(self, modelPath, mode = "cpu"):
        try:
            self.mode = mode
            if mode == "cpu":
                self.model = torch.load(modelPath, map_location = "cpu")
            elif mode == "gpu":
                self.model = torch.load(modelPath).to("cuda")
            self.model.eval()

            self.normalize = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.25, 0.25, 0.25))
            ])
        except Exception as e:
            print("Failed to load model: {}".format(str(e)))

    def quantify(self, img_path: str) -> np.ndarray:
        try:
            im = self.normalize(Image.open(img_path).convert("RGB")).unsqueeze(0)
            with torch.no_grad(): 
                if self.mode == "cpu":
                    return np.array(self.model(im))
                elif self.mode == "gpu":
                    return np.array(self.model(im.to("cuda")).cpu())
        except Exception as e:
            print("Failed to quantify: {}".format(str(e)))
            return np.array([])

    def quantify_in_batches(self, img_paths: list) -> np.ndarray:
        try:
            im = torch.stack([self.normalize(Image.open(img_path).convert("RGB")) for img_path in img_paths], dim = 0)
            with torch.no_grad(): 
                if self.mode == "cpu":
                    return np.array(self.model(im))
                elif self.mode == "gpu":
                    return np.array(self.model(im.to("cuda")).cpu())
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

modelPath = "D:/AdminFiles/Download/ckpt_epoch_50 (1).pth"
img1p = "D:/test2/11.jpg"
img2p = "D:/test2/10.jpg"
img3p = "D:/test2/9.jpg"

emo = EmoSame(modelPath)

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
