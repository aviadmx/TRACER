import cv2
import glob
import torch
import numpy as np
import albumentations as albu
from pathlib import Path
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split
from util.utils import get_files_recursive


class DatasetGenerate(Dataset):
    def __init__(self, img_folder, gt_folder, edge_folder, phase: str = 'train', transform=None, seed=None, split=True):
        self.images = sorted(get_files_recursive(img_folder, ext=["jpg", "jpeg", "png", "cr2", "webp", "tiff", 'tif']))
        self.gts = sorted(get_files_recursive(gt_folder, ext=["jpg", "jpeg", "png", "cr2", "webp", "tiff", 'tif']))
        self.edges = sorted(get_files_recursive(edge_folder, ext=["jpg", "jpeg", "png", "cr2", "webp", "tiff", 'tif']))
        self.transform = transform

        assert len(self.images) == len(self.gts) == len(
            self.edges), 'Amount of images, GT masks or edge masks is not equal'
        if split:
            train_images, val_images, train_gts, val_gts, train_edges, val_edges = train_test_split(self.images,
                                                                                                    self.gts,
                                                                                                    self.edges,
                                                                                                    test_size=0.05,
                                                                                                    random_state=seed)
            if phase == 'train':
                self.images = train_images
                self.gts = train_gts
                self.edges = train_edges
            elif phase == 'val':
                self.images = val_images
                self.gts = val_gts
                self.edges = val_edges
            else:  # Testset
                pass

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.gts[idx])
        if 'bgr_2021' in self.gts[0]:
            mask = np.mean(mask, 2)
        else:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        edge = cv2.imread(self.edges[idx])
        edge = cv2.cvtColor(edge, cv2.COLOR_BGR2GRAY)

        if self.transform is not None:
            augmented = self.transform(image=image, masks=[mask, edge])
            image = augmented['image']
            mask = np.expand_dims(augmented['masks'][0], axis=0)  # (1, H, W)
            mask = mask / 255.0
            edge = np.expand_dims(augmented['masks'][1], axis=0)  # (1, H, W)
            edge = edge / 255.0

        return image, mask, edge

    def __len__(self):
        return len(self.images)


class Test_DatasetGenerate(Dataset):
    def __init__(self, img_folder, gt_folder, transform=None):
        self.images = sorted(glob.glob(img_folder + '/*'))
        self.gts = sorted(glob.glob(gt_folder + '/*'))
        self.transform = transform

    def __getitem__(self, idx):
        image_name = Path(self.images[idx]).stem
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, self.gts[idx], original_size, image_name

    def __len__(self):
        return len(self.images)


def get_loader(datasets, phase: str, batch_size, shuffle,
               num_workers, transform, seed=None, split=True):
    if phase == 'test':
        dataset_list = []
        for _, (img_folder, gt_folder) in datasets.iterrows():
            dataset_list.append(Test_DatasetGenerate(img_folder, gt_folder, transform))
            if len(datasets) > 1:
                dataset = ConcatDataset(dataset_list)
            else:
                dataset = dataset_list[0]
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    else:
        dataset_list = []
        for _, (img_folder, gt_folder, edge_folder) in datasets.items():
            dataset_list.append(DatasetGenerate(img_folder, gt_folder, edge_folder, phase, transform, seed, split))
        if len(datasets) > 1:
            dataset = ConcatDataset(dataset_list)
        else:
            dataset = dataset_list[0]
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                 drop_last=True)

    print(f'{phase} length : {len(dataset)}')

    return data_loader


def get_train_augmentation(img_size, ver):
    if ver == 1:
        transforms = albu.Compose([
            albu.Resize(img_size, img_size, always_apply=True),
            albu.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    if ver == 2:
        transforms = albu.Compose([
            albu.OneOf([
                albu.HorizontalFlip(),
                albu.VerticalFlip(),
                albu.RandomRotate90()
            ], p=0.5),
            albu.OneOf([
                albu.RandomContrast(),
                albu.RandomGamma(),
                albu.RandomBrightness(),
            ], p=0.5),
            albu.OneOf([
                albu.MotionBlur(blur_limit=5),
                albu.MedianBlur(blur_limit=5),
                albu.GaussianBlur(blur_limit=5),
                albu.GaussNoise(var_limit=(5.0, 20.0)),
            ], p=0.5),
            albu.Resize(img_size, img_size, always_apply=True),
            albu.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    return transforms


def get_test_augmentation(img_size):
    transforms = albu.Compose([
        albu.Resize(img_size, img_size, always_apply=True),
        albu.Normalize([0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return transforms


def gt_to_tensor(gt):
    gt = cv2.imread(gt)
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY) / 255.0
    gt = np.where(gt > 0.5, 1.0, 0.0)
    gt = torch.tensor(gt, device='cuda', dtype=torch.float32)
    gt = gt.unsqueeze(0).unsqueeze(1)

    return gt
