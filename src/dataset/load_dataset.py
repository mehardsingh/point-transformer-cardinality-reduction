
from dataset_utils import *
from pathlib import Path
from tqdm import tqdm

def load_dataset(dataset_dir, max_points=None, sampled_points=1024):
    path = Path(dataset_dir)
    train_transforms = transforms.Compose([PointSampler(sampled_points), Normalize(), RandRotation_z(), RandomNoise(), ToTensor()])

    train_ds = PointCloudData(path, transform=train_transforms, max_points=max_points)
    valid_ds = PointCloudData(path, valid=True, folder='test', transform=train_transforms, max_points=max_points)

    return train_ds, valid_ds