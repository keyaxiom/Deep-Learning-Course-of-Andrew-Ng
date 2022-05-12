import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import torch
from PIL import Image
import torch.utils.data as Data


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_data(train, noise_type,  noisy_ratio):
    data = None
    labels = None
    label_1s = None
    if train == True:
        for i in range(1, 6):
            batch = unpickle('./data/cifar-10-batches-py/data_batch_' + str(i))
            if i == 1:
                data = batch[b'data']
            else:
                data = np.concatenate([data, batch[b'data']])
        #load noisy label
        noisy_label_root = './Cifar10_' + str(noise_type) + '_noisy_' + str(noisy_ratio) + 'P.npy'
        print(noisy_label_root)
        labels = np.load(noisy_label_root)

    else:
        batch = unpickle('./data/cifar-10-batches-py/test_batch')
        data = batch[b'data']
        labels = batch[b'labels']
    return data, labels


def target_transform(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target


transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    #transforms.RandomCrop(24),  
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


class Cifar10_Dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, noise_type=None, noisy_ratio=None):

        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if self.train:
            self.train_data, self.train_labels = get_data(train, noise_type, noisy_ratio)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))

            self.train_data = self.train_data.transpose((0, 2, 3, 1))
            
        else:
            self.test_data, self.test_labels = get_data(train, noise_type, noisy_ratio)
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))
        pass

    def __getitem__(self, index):

        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
        else:
            img, label = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(label)

        return img, target

    def __len__(self):

        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


if __name__ == '__main__':
    data, labels = get_data(train=True, noise_type='flip',  noisy_ratio='30')
    print(labels.shape)
