import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dicts = pickle.load(fo, encoding='bytes')
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
        noisy_label_root = './Cifar10_' + str(noise_type) + '_noisy_' + str(noisy_ratio) + 'P.npy'
        print(noisy_label_root)
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


transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


class Cifar10Dataset(Dataset):
    def __init__(self, _type='train', _transform=None, _target_transform=None, _noise_type=None, _noisy_ratio=None):
        assert _type in ['train', 'val', 'test']

        self.transform = _transform
        self.target_transform = _target_transform
        self.type = _type

        if self.type == 'train':
            self.train_data, self.train_true_labels, self.train_noise_labels = get_data(True, _noise_type, _noisy_ratio)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))

            self.train_data = self.train_data.transpose((0, 2, 3, 1))
            
        elif self.type == 'val':
            self.test_data, self.test_true_labels, _ = get_data(False, _noise_type, _noisy_ratio)
            self.test_data = self.test_data[0:1000].reshape((1000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))

        elif self.type == 'test':
            self.test_data, self.test_true_labels, _ = get_data(False, _noise_type, _noisy_ratio)
            self.test_data = self.test_data[1000:].reshape((9000, 3, 32, 32))
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


if __name__ == '__main__':
    train_data, true_label, noise_label = get_data(train='True', noise_type='uniform', noisy_ratio='30')
    print(np.sum(true_label==noise_label)/50000)
