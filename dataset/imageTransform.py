import torch
import PIL
from PIL import Image
import numpy as np


class ImageTransform():
    
    def __init__():
        pass

    @classmethod
    def load(cls, path: str) -> PIL.Image:
        return Image.open(path)

    @classmethod
    def crop(cls, img: PIL.Image) -> PIL.Image:
        w, h = img.size
        # crop center
        center_w = w / 2
        center_h = h / 2

        if(w > h):
            img_crop = img.crop((center_w - center_h, 0, center_w + center_h, h))
        else:
            img_crop = img.crop((0, center_h - center_w, w, center_h + center_w))

        return img_crop

    @classmethod
    def resize(cls, img: PIL.Image, size: tuple = (224,224)) -> PIL.Image:
        return img.resize(size)

    @classmethod
    def toTensor(cls, pic: PIL.Image) -> torch.tensor:
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.float().div(self.norm_value)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)

        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()

        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img

