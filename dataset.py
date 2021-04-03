import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
import sys
import pickle
import os
import numpy as np
def get_full_supervised(root, meta_path, train=True, **kwargs):
    meta_path = os.path.join (meta_path, "fully-supervised-split.csv")
    # read
    files = set()
    with open(meta_path, "r") as f:
        f.readline()
        for line in f:
            parts = line.strip().split(",")
            _is_train = bool(int(parts[1].strip()))
            if train == _is_train:
                files.add(parts[0].strip())

    dataset = torchvision.datasets.CIFAR10(root, train=train, **kwargs)
    if train :
        downloaded_list = dataset.train_list
    else :
        downloaded_list = dataset.test_list
    base_folder = 'cifar-10-batches-py'
    tot_filenames = _get_filenames (root, base_folder, downloaded_list)
    sub_indices = [i for i in range(len(tot_filenames)) if tot_filenames[i] in files]
    sub_data = torch.utils.data.Subset(dataset, sub_indices)
    print("==>Only {}/{} original data".format(len(sub_indices), len(tot_filenames)))
    return sub_data

def _get_filenames(root, base_folder, downloaded_list):
    filenames = []
    for file_name, checksum in downloaded_list :
        file_path = os.path.join (root, base_folder, file_name)
        with open (file_path, 'rb') as f :
            if sys.version_info[0] == 2 :
                entry = pickle.load (f)
            else :
                entry = pickle.load (f, encoding='latin1')
            filenames.extend (entry['filenames'])
    return filenames


class CIFAR10_BAG(torchvision.datasets.CIFAR10):
    def __init__(self, root, meta_path, trial=0, train=True, transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_BAG, self).__init__(root, train=train, transform=transform, target_transform=target_transform,
                 download=download)
        assert trial >= 0 and trial < 5, trial
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.filenames = _get_filenames(self.root, self.base_folder, downloaded_list)
        # now load the picked numpy arrays

        if train:
            meta_path = os.path.join(meta_path, "train.trial={}.csv".format(trial))
        else:
            meta_path = os.path.join(meta_path, "test.trial={}.csv".format(trial))

        self.samples = self.__load_meta(meta_path)
        self.filename_to_idx = dict(zip(self.filenames, range(len(self.filenames))))

    def __load_meta(self, path):
        samples = []
        with open(path, "r") as f:
            f.readline()
            for line in f:
                line = line.strip().split(",")
                instances = line[0].strip().split(" ")
                cnts = [int(i) for i in line[1].strip().split(" ")]
                samples.append((instances, cnts))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data = []
        instance_target = []
        bag_image_names, counts = self.samples[index]
        
        counts = torch.tensor(counts).long()
        for name in bag_image_names:
            img = self.data[self.filename_to_idx[name]]
            instance_target.append(self.targets[self.filename_to_idx[name]])
            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)
            data.append(img.unsqueeze(0))
        data = torch.cat(data, dim=0)  # (b, 3, H, W)
        instance_target = torch.tensor(instance_target).long()
        # data (bag, channel, h, w)
        # counts (bag, n_class)
        # instance_target: (bag, )
        return data, counts, instance_target


if __name__ == '__main__':
    """test"""
    import torchvision.transforms as transforms
    transform_train = transforms.Compose ([
        transforms.RandomCrop (32, padding=4),
        transforms.RandomHorizontalFlip (),
        transforms.ToTensor (),
        transforms.Normalize ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = CIFAR10_BAG(root='./data', meta_path="./meta/reuse.n=9000.bag=8", train=True, download=True, transform=transform_train)
    assert trainset[0][0].shape == (8, 3, 32, 32),  trainset[0][0].shape
    subset = get_full_supervised(root='./data', meta_path="./meta/reuse.n=9000.bag=8", train=True, download=True, transform=transform_train)
