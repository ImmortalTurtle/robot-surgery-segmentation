"""
Script generates predictions, splitting original images into tiles, and assembling prediction back together
"""
import argparse
from prepare_data import get_split
from dataset import CustomDataset
import cv2
from models import UNet16, LinkNet34, UNet11, UNet, AlbuNet
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
import utils
from torch.utils.data import DataLoader
from torch.nn import functional as F
from albumentations import Compose, Normalize

from dataset import Rescale, RandomCrop


def img_transform(p=1):
    return Compose([
        Normalize(p=1)
    ], p=p)


def get_model(model_path, model_type='UNet11', problem_type='binary'):
    """

    :param model_path:
    :param model_type: 'UNet', 'UNet16', 'UNet11', 'LinkNet34', 'AlbuNet'
    :param problem_type: 'binary', 'parts', 'instruments'
    :return:
    """
    if problem_type == 'binary':
        num_classes = 1
    elif problem_type == 'parts':
        num_classes = 4
    elif problem_type == 'instruments':
        num_classes = 8

    if model_type == 'UNet16':
        model = UNet16(num_classes=num_classes)
    elif model_type == 'UNet11':
        model = UNet11(num_classes=num_classes)
    elif model_type == 'LinkNet34':
        model = LinkNet34(num_classes=num_classes)
    elif model_type == 'AlbuNet':
        model = AlbuNet(num_classes=num_classes)
    elif model_type == 'UNet':
        model = UNet(num_classes=num_classes)

    state = torch.load(str(model_path))
    state = {key.replace('module.', ''): value for key, value in state['model'].items()}
    model.load_state_dict(state)

    if torch.cuda.is_available():
        return model.cuda()

    model.eval()

    return model


def predict(model, from_file_names, batch_size, to_path, problem_type, img_transform):
    loader = DataLoader(
        dataset=CustomDataset(from_file_names, transform=img_transform, mode='predict', problem_type=problem_type),
        shuffle=False,
        batch_size=1,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available()
    )

    with torch.no_grad():
        for batch_num, (inputs, paths) in enumerate(tqdm(loader, desc='Predict')):
            inputs = utils.cuda(inputs)
            outputs = model(inputs)

            for i, image_name in enumerate(paths):
                factor = 255
                t_mask = (F.sigmoid(outputs[i, 0]).data.cpu().numpy() * factor).astype(np.uint8)

                h, w = t_mask.shape

                instrument_folder = Path(paths[i]).parent.parent.name

                (to_path / instrument_folder).mkdir(exist_ok=True, parents=True)
                (to_path / instrument_folder / 'overlayed').mkdir(exist_ok=True, parents=True)
                img = inputs.cpu().numpy()[0]
                t_mask = np.repeat(t_mask[:,:,np.newaxis], 3, 2)

                img = img*255/np.max(img)
                img = img.T.astype(np.uint8)
                img = np.swapaxes(img, 0, 1)
                img = cv2.addWeighted(img, 0.7, t_mask, 0.3, 1)
                cv2.imwrite(str(to_path / instrument_folder / (Path(paths[i]).stem + '.png')), t_mask)
                cv2.imwrite(str(to_path / instrument_folder / 'overlayed'/ (Path(paths[i]).stem + '.png')), img[:,:,::-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model_path', type=str, default='data/models/UNet', help='path to model folder')
    arg('--model_type', type=str, default='UNet', help='network architecture',
        choices=['UNet', 'UNet11', 'UNet16', 'LinkNet34', 'AlbuNet'])
    arg('--output_path', type=str, help='path to save images', default='1')
    arg('--batch-size', type=int, default=4)
    arg('--fold', type=int, default=-1, choices=[0, 1, 2, 3, -1], help='-1: all folds')
    arg('--problem_type', type=str, default='binary', choices=['binary', 'parts', 'instruments'])
    arg('--workers', type=int, default=12)

    args = parser.parse_args()

    _, file_names = get_split()
    model = get_model(str(Path(args.model_path).joinpath('model_{fold}.pt'.format(fold=fold))),
                        model_type=args.model_type, problem_type=args.problem_type)

    print('num file_names = {}'.format(len(file_names)))

    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    predict(model, file_names, args.batch_size, output_path, problem_type=args.problem_type,
            img_transform=img_transform(p=1))