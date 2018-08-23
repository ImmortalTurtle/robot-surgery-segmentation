from pathlib import Path
import argparse
import cv2
import numpy as np
from tqdm import tqdm

height, width, h_start, w_start = 0


def general_dice(y_true, y_pred):
    result = []

    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    for instrument_id in set(y_true.flatten()):
        if instrument_id == 0:
            continue
        result += [dice(y_true == instrument_id, y_pred == instrument_id)]

    return np.mean(result)


def general_jaccard(y_true, y_pred):
    result = []

    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    for instrument_id in set(y_true.flatten()):
        if instrument_id == 0:
            continue
        result += [jaccard(y_true == instrument_id, y_pred == instrument_id)]

    return np.mean(result)


def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)


def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--train_path', type=str, default='data/train',
        help='path where train images with ground truth are located')
    arg('--target_path', type=str, default='predictions/unet11', help='path with predictions')
    args = parser.parse_args()
    
    for file_name in (Path(args.train_path) / 'masks').glob('*'):
        y_true = (cv2.imread(str(file_name), 0) > 0).astype(np.uint8)

        pred_file_name = Path(args.target_path) / file_name.name

        y_pred = (cv2.imread(str(pred_file_name), 0) > 255 * 0.5).astype(np.uint8)[h_start:h_start + height,
                    w_start:w_start + width]

        result_dice = dice(y_true, y_pred)
        result_jaccard = jaccard(y_true, y_pred)

    
    print('Dice = ',result_dice)
    print('Jaccard = ', result_jaccard)
