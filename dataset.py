import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from albumentations.torch.functional import img_to_tensor
from skimage import transform


class CustomDataset(Dataset):
    def __init__(self, file_names: list, to_augment=False, transform=None, mode='train'):
        self.file_names = file_names
        self.to_augment = to_augment
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]
        image = load_image(img_file_name)
        mask = load_mask(img_file_name)

        data = {"image": image, "mask": mask}
        augmented = self.transform(**data)
        image, mask = augmented["image"], augmented["mask"]

        if self.mode == 'train':
            return img_to_tensor(image), torch.from_numpy(np.expand_dims(mask, 0)).float()
        else:
            return img_to_tensor(image), str(img_file_name)


def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path):
    mask_folder = 'masks'
    factor = 255

    mask = cv2.imread(str(path).replace('images', mask_folder).replace('jpg', 'png'), 0)

    return (mask / factor).astype(np.uint8)


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image, mask):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint((h - new_h) / 2, (h - new_h) / 8) if new_h < h else 0
        left = np.random.randint((w - new_w) / 2, (w - new_w) / 8) if new_w < w else 0
        top, left = np.clip(top, 0, h - new_h), np.clip(left, 0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        mask = mask[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'mask': mask}

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image, mask):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        img = (img * np.max(image) / np.max(img)).astype(np.uint8)
        mask = transform.resize(mask, (new_h, new_w))
        mask = (mask / np.max(mask)).astype(np.uint8)

        return {'image': img, 'mask': mask}