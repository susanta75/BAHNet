import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
import cv2
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2


class NormalDataset(Dataset):
    def __init__(self, data_path, transform, mode='train'):
        self.imgs_list = sorted(glob(data_path + "/image/*"))
        self.masks_list = sorted(glob(data_path + "/mask/*"))
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, index):
        img_path = self.imgs_list[index]
        mask_path = self.masks_list[index]
        mask_name = mask_path.split("/")[-1]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.mode != 'train':
            ori_mask = torch.from_numpy(mask).float() / 255.0

        augmented = self.transform(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask'].float() / 255.0

        if self.mode == 'train':
            return {'img': img, 'mask': mask}
        else:
            return {'img': img, 'mask': mask, 'ori_mask': ori_mask, 'mask_name': mask_name}


def get_augmentation(version=0, img_size=512):
    if version == 0:  # Training augmentation
        transforms = albu.Compose([
            # --- Scale jitter ---
            # Polyps appear at very different sizes across datasets.
            # Kvasir polyps are often large; ETIS-Lari / CVC-ColonDB polyps
            # are systematically smaller.  RandomResizedCrop forces the model
            # to segment polyps that fill anywhere from 25% to 100% of the
            # FOV, so it never over-fits to one scale distribution.
            # p=0.8 so 80% of images get scale-jittered; the trailing Resize
            # handles the remaining 20%.
            albu.RandomResizedCrop(
                size=(img_size, img_size),
                scale=(0.5, 1.0), ratio=(0.75, 1.33),
                p=0.8
            ),
            # --- Geometric augmentation ---
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            # --- Color / appearance augmentation ---
            # Different colonoscopes and processing pipelines produce very
            # different brightness, contrast and white-balance.  Without this,
            # the model latches onto Kvasir's specific color palette as a
            # polyp cue, which fails on CVC-ColonDB / ETIS-Lari.
            albu.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            albu.Rotate(limit=30, p=0.5),
            # Safety resize: guarantees img_size output for the 20% that
            # skipped RandomResizedCrop, and is a no-op for the 80% that
            # were already resized by it.
            albu.Resize(img_size, img_size),
            albu.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:  # Testing augmentation — deterministic, no jitter
        transforms = albu.Compose([
            albu.Resize(img_size, img_size),
            albu.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    return transforms


def getSODDataloader(data_path, batch_size, num_workers, mode, img_size=512):
    if mode == "train":
        transform = get_augmentation(0, img_size)
    else:
        transform = get_augmentation(1, img_size)

    dataset = NormalDataset(data_path + "/" + mode, transform, mode)

    dataLoader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataLoader
