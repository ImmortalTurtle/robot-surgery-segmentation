from pathlib import Path

data_path = Path('data/')
train_path = data_path / 'train'
SIZE=480

def get_split():
    train_path = data_path / 'train'
    val_path = data_path / 'val'

    train_file_names = list((train_path / 'images').glob('*'))
    val_file_names = list((val_path / 'images').glob('*'))

    return train_file_names, val_file_names