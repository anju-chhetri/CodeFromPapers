

import torch
import numpy as np
from torchvision import datasets
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from utils.cutout import Cutout
from torch.utils.data import Subset, Dataset
import random
import numpy as np

def read_dataset(pic_path, batch_size=16,valid_size=0.2,num_workers=0):
    """
    batch_size: Number of loaded drawings per batch
    valid_size: Percentage of training set to use as validation
    num_workers: Number of subprocesses to use for data loading
    pic_path: The path of the pictrues
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  
        transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        # Cutout(n_holes=1, length=16),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_data = datasets.CIFAR10(pic_path, train=True,
                                download=True, transform=transform_train)
    valid_data = datasets.CIFAR10(pic_path, train=True,
                                download=True, transform=transform_test)
    test_data = datasets.CIFAR10(pic_path, train=False,
                                download=True, transform=transform_test)
        
    num_train = len(train_data)
    indices = list(range(num_train))
    # random indices
    np.random.shuffle(indices)
    # the ratio of split
    split = int(np.floor(valid_size * num_train))
    # divide data to radin_data and valid_data
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    # 无放回地按照给定的索引列表采样样本元素
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, 
        sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
        num_workers=num_workers, shuffle=True)

    return train_loader,valid_loader,test_loader

def read_cifar100_dataset(pic_path, batch_size=16,valid_size=0.2,num_workers=0):
    """
    batch_size: Number of loaded drawings per batch
    valid_size: Percentage of training set to use as validation
    num_workers: Number of subprocesses to use for data loading
    pic_path: The path of the pictrues
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  
        transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        # Cutout(n_holes=1, length=16),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_data = datasets.CIFAR100(pic_path, train=True,
                                download=True, transform=transform_train)
    valid_data = datasets.CIFAR100(pic_path, train=True,
                                download=True, transform=transform_test)
    test_data = datasets.CIFAR100(pic_path, train=False,
                                download=True, transform=transform_test)
        
    num_train = len(train_data)
    indices = list(range(num_train))
    # random indices
    np.random.shuffle(indices)
    # the ratio of split
    split = int(np.floor(valid_size * num_train))
    # divide data to radin_data and valid_data
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    # 无放回地按照给定的索引列表采样样本元素
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size,
        sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers)

    return train_loader,valid_loader,test_loader


def read_mnist_dataset(pic_path, batch_size=16,valid_size=0.2,num_workers=0):
    """
    batch_size: Number of loaded drawings per batch
    valid_size: Percentage of training set to use as validation
    num_workers: Number of subprocesses to use for data loading
    pic_path: The path of the pictrues
    """

    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to 32x32
        transforms.ToTensor(),  # Convert to tensor and normalize to [0,1]
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x)  # Convert to RGB
    ])

    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to 32x32
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Convert to tensor and normalize to [0,1]
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x)  # Convert to RGB
    ])

    train_data = datasets.MNIST(pic_path, train=True,
                                download=True, transform=transform)
    valid_data = datasets.MNIST(pic_path, train=True,
                                download=True, transform=transform_test)
    test_data = datasets.MNIST(pic_path, train=False,
                                download=True, transform=transform_test)
        
    num_train = len(train_data)
    indices = list(range(num_train))
    # random indices
    np.random.shuffle(indices)
    # the ratio of split
    split = int(np.floor(valid_size * num_train))
    # divide data to radin_data and valid_data
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    # 无放回地按照给定的索引列表采样样本元素
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, 
        sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
        num_workers=num_workers)

    return train_loader,valid_loader,test_loader


def read_fashion_mnist_dataset(pic_path, batch_size=16,valid_size=0.2,num_workers=0):
    """
    batch_size: Number of loaded drawings per batch
    valid_size: Percentage of training set to use as validation
    num_workers: Number of subprocesses to use for data loading
    pic_path: The path of the pictrues
    """

    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to 32x32
        transforms.ToTensor(),  # Convert to tensor and normalize to [0,1]
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x)  # Convert to RGB
    ])

    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to 32x32
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Convert to tensor and normalize to [0,1]
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x)  # Convert to RGB
    ])

    train_data = datasets.FashionMNIST(pic_path, train=True,
                                download=True, transform=transform)
    valid_data = datasets.FashionMNIST(pic_path, train=True,
                                download=True, transform=transform_test)
    test_data = datasets.FashionMNIST(pic_path, train=False,
                                download=True, transform=transform_test)
        
    num_train = len(train_data)
    indices = list(range(num_train))
    # random indices
    np.random.shuffle(indices)
    # the ratio of split
    split = int(np.floor(valid_size * num_train))
    # divide data to radin_data and valid_data
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    # 无放回地按照给定的索引列表采样样本元素
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, 
        sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
        num_workers=num_workers)

    return train_loader,valid_loader,test_loader


def read_ood_data(pic_path, batch_size=16,valid_size=0.2,num_workers=0):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  
        transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        # Cutout(n_holes=1, length=16),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_data = datasets.SVHN(pic_path, split='train',
                                download=True, transform=transform_train)
    test_data = datasets.SVHN(pic_path, split='test',
                                download=True, transform=transform_test)
    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
        shuffle=True, num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers)

    return train_loader, test_loader

class LSUNDataset(Dataset):
    def __init__(self, data_tensor, labels_tensor):
        self.data = data_tensor
        self.labels = labels_tensor
        self.transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor()])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform:
            self.data[idx]=self.transform(self.data[idx])
        return self.data[idx], self.labels[idx]

def read_lsun_data(pic_path, batch_size=16,valid_size=0.2,num_workers=0):

    train_data_loaded = np.load('/path/to/train_data.npy')
    train_labels_loaded = np.load('/path/to/train_label.npy')    # train_data_l = []
    val_data_loaded = np.load('/path/to/val_data.npy')
    val_labels_loaded = np.load('/path/to/val_label.npy')

    train_data_tensor = torch.tensor(train_data_loaded)
    train_labels_tensor = torch.tensor(train_labels_loaded)
    val_data_tensor = torch.tensor(val_data_loaded)
    val_labels_tensor = torch.tensor(val_labels_loaded)

    train_data = LSUNDataset(train_data_tensor, train_labels_tensor)
    val_data = LSUNDataset(val_data_tensor, val_labels_tensor) 
   # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, 
        shuffle=True,  num_workers=num_workers)
    

    # test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, val_loader



def mnist(pic_path, batch_size=16, valid_size=0.2, num_workers=0):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to 32x32
        transforms.ToTensor(),  # Convert to tensor and normalize to [0,1]
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x)  # Convert to RGB
    ])


import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class textures(Dataset):
    def __init__(self, img_dir, transform=None):
        """
        Args:
            img_dir (str): Path to the images directory.
            labels_file (str): Path to the labels file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_dir = img_dir
        self.image_files = os.listdir(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        
        # Open the image
        image = Image.open(img_path).convert("RGB")
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        
        return image, 0

def read_textures_data(pic_path, batch_size=16,valid_size=0.2,num_workers=0):

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Directory and file paths
    img_dir = '/path/to/texture_dataset/'

    # Create dataset and dataloader
    dataset = textures(img_dir=img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    return dataloader

def read_places365_dataset(pic_path, batch_size=16,valid_size=0.2,num_workers=0):
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),          # Randomly crop to 224x224 for data augmentation
        transforms.RandomHorizontalFlip(p=0.5),     # Random horizontal flip with a 50% chance
        transforms.ToTensor(),                      # Convert to tensor
    ])

    # Testing (validation) transforms
    # test_transform = transforms.Compose([
    #     transforms.Resize(256),                     # Resize the shorter side to 256 pixels
    #     transforms.CenterCrop(224),                 # Center crop to 224x224
    #     transforms.ToTensor(),                      # Convert to tensor
    # ])
    test_transform= torchvision.models.VGG16_Weights.IMAGENET1K_V1.transforms()

    places365_data = datasets.Places365(pic_path, split = 'train-standard', download=False, transform=train_transform)
    places365_val_data = datasets.Places365(pic_path, split="val", download=False, transform=test_transform)
    train_loader = torch.utils.data.DataLoader(places365_data,
                                                batch_size=batch_size,
                                                shuffle=True,                                          
                                                num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(places365_val_data,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    
    return train_loader, val_loader


import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import VGG16_Weights

# Step 1: Define the transform using VGG16 weights
weights = VGG16_Weights.IMAGENET1K_V1
inference_transform = weights.transforms()

# Step 2: Create a custom dataset class for unlabeled images
class UnlabeledImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")  # Convert to RGB

        if self.transform:
            image = self.transform(image)

        return image, 0  # Return only the image tensor


def read_imagenet_dataset(pic_path, batch_size=16,valid_size=0.2,num_workers=0):
    
    inference_transform= torchvision.models.ResNet50_Weights.DEFAULT.transforms()

    imagenet_val_dataset = UnlabeledImageDataset(os.path.join(pic_path, "imagenet_val_small"), transform=inference_transform)
    imagenet_val_loader = DataLoader(
        dataset=imagenet_val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Shuffling is not needed for validation
        num_workers=num_workers,  # Adjust num_workers based on your CPU capability
        pin_memory=True  # Enable if using a CUDA device
    )

    return imagenet_val_loader


def read_inaturlist_dataset(pic_path, batch_size=16,valid_size=0.2,num_workers=0):
    inference_transform= torchvision.models.ResNet50_Weights.DEFAULT.transforms()

    natural_data= torchvision.datasets.INaturalist(pic_path, download=True, transforms=inference_transform, version="2021_valid")

    natural_val_loader = DataLoader(
        dataset=natural_data,
        batch_size=batch_size,
        shuffle=False,  # Shuffling is not needed for validation
        num_workers=num_workers,  # Adjust num_workers based on your CPU capability
        pin_memory=True  # Enable if using a CUDA device
    )

    return natural_val_loader







import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
from sklearn.model_selection import train_test_split

class Kvasirv2Dataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def read_kvsair_dataset(pic_path, batch_size=16,valid_size=0.2,num_workers=0):
    # Data augmentation and normalization for images
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization values for ImageNet
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization values for ImageNet
    ])

    # Split the dataset into training and validation sets
    id_images = []  # List to hold in-distribution image paths
    ood_images = []  # List to hold out-of-distribution image paths
    id_labels = []
    ood_labels = []  # List to hold out-of-distribution labels

    # Populate lists with image paths
    for class_idx, class_name in enumerate(os.listdir(f'{pic_path}/kvsair/ID')):
        class_path = os.path.join(f'{pic_path}/kvsair/ID', class_name)
        for img_name in os.listdir(class_path):
            id_images.append(os.path.join(class_path, img_name))
            id_labels.append(class_idx)

    for class_name in os.listdir(f'{pic_path}/kvsair/OOD'):
        class_path = os.path.join(f'{pic_path}/kvsair/OOD', class_name)
        for img_name in os.listdir(class_path):
            ood_images.append(os.path.join(class_path, img_name))
            ood_labels.append(0)  # Assign a label of 0 to all OOD images

    # Randomly split the ID images for training (2400) and validation (600)
    train_id, val_id, train_labels, val_labels= train_test_split(id_images, id_labels, train_size=2400, test_size=600)

    # Randomly select 5000 OOD images for testing
    ood_images_selected = random.sample(list(zip(ood_images, ood_labels)), 5000)
    ood_images_selected, ood_labels_selected = zip(*ood_images_selected)
# Create datasets
    train_dataset = Kvasirv2Dataset(image_paths=train_id, labels=train_labels, transform=transform_train)
    val_dataset = Kvasirv2Dataset(image_paths=val_id, labels=val_labels, transform=transform_test)
    test_dataset = Kvasirv2Dataset(image_paths=ood_images_selected, labels=ood_labels_selected, transform=transform_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, val_loader, test_loader






class GastrovisionDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def read_gastrovision_dataset(pic_path, batch_size=16,valid_size=0.2,num_workers=0):
    # Data augmentation and normalization for images
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization values for ImageNet
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization values for ImageNet
    ])

    # Split the dataset into training and validation sets
    id_images = []  # List to hold in-distribution image paths
    ood_images = []  # List to hold out-of-distribution image paths
    id_labels = []
    ood_labels = []  # List to hold out-of-distribution labels

    # Populate lists with image paths
    for class_idx, class_name in enumerate(os.listdir(f'{pic_path}/Gastrovision/ID')):
        class_path = os.path.join(f'{pic_path}/Gastrovision/ID', class_name)
        for img_name in os.listdir(class_path):
            id_images.append(os.path.join(class_path, img_name))
            id_labels.append(class_idx)

    for class_name in os.listdir(f'{pic_path}/Gastrovision/OOD'):
        class_path = os.path.join(f'{pic_path}/Gastrovision/OOD', class_name)
        for img_name in os.listdir(class_path):
            ood_images.append(os.path.join(class_path, img_name))
            ood_labels.append(0)  # Assign a label of 0 to all OOD images

    # Randomly split the ID images for training (2400) and validation (600)
    train_id, val_id, train_labels, val_labels= train_test_split(id_images, id_labels, train_size=3804, test_size=955)

    # Randomly select 5000 OOD images for testing
    ood_images_selected = random.sample(list(zip(ood_images, ood_labels)), 3241)
    ood_images_selected, ood_labels_selected = zip(*ood_images_selected)
# Create datasets
    train_dataset = GastrovisionDataset(image_paths=train_id, labels=train_labels, transform=transform_train)
    val_dataset = GastrovisionDataset(image_paths=val_id, labels=val_labels, transform=transform_test)
    test_dataset = GastrovisionDataset(image_paths=ood_images_selected, labels=ood_labels_selected, transform=transform_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, val_loader, test_loader