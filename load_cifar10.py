import torch.utils.data
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def unpickle(file):
    import pickle
    with open(file, 'rb') as f:
        dicts = pickle.load(f, encoding='bytes')
    return dicts


def get_data(train, noise_type, noisy_ratio):
    data = None
    true_labels = None
    noise_labels = None

    if train:
        for i in range(1, 6):
            batch = unpickle('./data/cifar-10-batches-py/data_batch_' + str(i))
            if i == 1:
                data = batch[b'data']
                true_labels = batch[b'labels']
            else:
                data = np.concatenate([data, batch[b'data']])
                true_labels = np.concatenate([true_labels, batch[b'labels']])
        # load noisy label
        noisy_label_root = './noisy_labels/Cifar10_' + str(noise_type) + '_noisy_' + str(
            noisy_ratio) + 'P.npy'
        noise_labels = np.load(noisy_label_root)
    else:
        batch = unpickle('./data/cifar-10-batches-py/test_batch')
        data = batch[b'data']
        true_labels = batch[b'labels']

    return data, true_labels, noise_labels


def target_transform(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target


class NoisyCifar10(Dataset):
    def __init__(self, _type='train', _transform=None, _target_transform=target_transform,
                 _noise_type=None, _noisy_ratio=None):
        assert _type in ['train', 'val', 'test']

        self.transform = _transform
        self.target_transform = _target_transform
        self.type = _type

        if self.type == 'train':
            self.train_data, self.train_true_labels, self.train_noise_labels = get_data(True, _noise_type, _noisy_ratio)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))

        elif self.type == 'test':
            self.test_data, self.test_true_labels, _ = get_data(False, _noise_type, _noisy_ratio)
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):

        if self.type == 'train':
            true_target = None
            noise_target = None
            img, _true_label, _noise_label = self.train_data[index], self.train_true_labels[index], \
                                             self.train_noise_labels[index]
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                true_target = self.target_transform(_true_label)
                noise_target = self.target_transform(_noise_label)

            return img, true_target, noise_target

        else:
            target = None
            img, label = self.test_data[index], self.test_true_labels[index]
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(label)

            return img, target

    def __len__(self):

        if self.type == 'train':
            return len(self.train_data)
        else:
            return len(self.test_data)


