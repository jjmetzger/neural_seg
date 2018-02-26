import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from skimage import io
import re, os
from os.path import join
from fc_densenet_torch import fcdensenet_tiny, fcdensenet56, fcdensenet103
import platform
from affine_transforms import CustomRandomTransform


def run(base_dir, model_type='56', batch_size=4, shuffle=True, n_epoch=2, weight=None, transform=True, lr=0.0001):

    cuda_available = torch.cuda.is_available()

    # if platform.system() == 'Darwin':
    #     base_dir = '/Users/jakob/Documents/RU/Code/FC-DenseNet'
    # elif platform.system() == 'Linux':
    #     # base_dir = '/home/jakob/Code/nn/FC-DenseNet/data'
    #     # base_dir = '/home/jakob/Code/nn/lumen/data/ground_truth_seg/'
    #     base_dir = '/home/jakob/Code/nn/lumen/PAX6/'
    # else:
    #     raise RuntimeError('OS unknown')

    lumen_path_train = join(base_dir, 'train/')
    trainloader = dataloader(lumen_path_train, batch_size=batch_size, shuffle=shuffle, transform=transform)
    # criterion = nn.CrossEntropyLoss()

    loss = nn.NLLLoss2d(weight=weight)
    #loss = nn.NLLLoss2d()

    if model_type == '103':
        net = fcdensenet103()
    elif model_type == '56':
        net = fcdensenet56()
    elif model_type == 'tiny':
        net = fcdensenet_tiny()
    else:
        raise NotImplementedError('unkonwn network architecture')

    if cuda_available:
        net.cuda()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    train(net, trainloader, loss, optimizer, n_epoch=n_epoch)

    return net


def save_net(net, filename):
    torch.save(net.state_dict(), filename)


def dataloader(path, batch_size=4, shuffle=True, num_workers=1, transform=True):

    if transform:
        mytransform = CustomRandomTransform()
    else:
        mytransform = None

    lumen_ds = LumenDataset(path, transform=mytransform)
    return DataLoader(lumen_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def train(net, trainloader, criterion, optimizer, n_epoch=2):

    cuda_available = torch.cuda.is_available()

    for epoch in range(n_epoch):  # loop over the dataset multiple times
        # print('EPOCH', epoch)
        # print('-------------------------')
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data['image'], data['image_labeled']

            # wrap them in Variable
            if cuda_available:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            # print('loss:', loss.data[0])

            print_every_n = 2
            if i % print_every_n == print_every_n-1:  # print every 2000 mini-batches
                print('[%d/%d, %5d] loss: %.3f' %
                      (epoch + 1, n_epoch, i + 1, running_loss / print_every_n))
                running_loss = 0.0

    print('Finished Training')


class LumenDataset(Dataset):
    """Lumen dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        p = re.compile('\d+\.tif')

        self.filenames = []
        self.filenames_labeled = []
        for file in os.listdir(self.root_dir):
            if p.findall(file):
                self.filenames.append(join(self.root_dir, file))
                self.filenames_labeled.append(
                    # join(self.root_dir, os.path.splitext(file)[0] + '_labeled' + os.path.splitext(file)[1]))
                    join(self.root_dir, os.path.splitext(file)[0] + '_Labels' + os.path.splitext(file)[1]))

        # print(self.filenames)
        assert len(self.filenames) == len(self.filenames_labeled)
        self.n_samples = len(self.filenames)
        if self.n_samples == 0:
            print('WARNING: no images found')
        else:
            print('found', self.n_samples, 'files')

        self.im_shape = io.imread(self.filenames[0]).shape

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # print(idx)
        image = io.imread(self.filenames[idx])
        image_labeled = io.imread(self.filenames_labeled[idx])

        image = (image/2**8).astype(np.uint8)
        image_labeled = (image_labeled*255).astype(np.uint8)

        if self.transform:
            image, image_labeled = self.transform(image, image_labeled)

        # normalize
        image = (image - image.mean()) / image.std()
        image_labeled = (image_labeled > 100).astype(np.long)
        #         image_labeled = (image_labeled - image_labeled.mean())/image_labeled.std()        # DONT on labels!!!


        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = np.tile(image, (3, 1, 1))
        image = np.expand_dims(image, 0) #one channel, needs explicit dimension

        #         image = image.transpose((2, 0, 1))
        #         image_labeled = np.tile(image_labeled, (3,1,1))
        #         image_labeled = image_labeled.transpose((2, 0, 1))
        image = image.astype(np.float32)
        # image_labeled = image_labeled.astype(np.long)

        return {'image': torch.from_numpy(image), 'image_labeled': torch.from_numpy(image_labeled)}


class LumenTestSet(LumenDataset):
    # for loading a dataset without ground truth lumen identification

    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        p = re.compile('\d+\.tif')

        self.filenames = []
        for file in os.listdir(self.root_dir):
            if p.findall(file):
                self.filenames.append(join(self.root_dir, file))

        self.n_samples = len(self.filenames)
        if self.n_samples == 0:
            print('WARNING: no images found')

        self.im_shape = io.imread(self.filenames[0]).shape


    def __getitem__(self, idx):
        # print(idx)
        image = io.imread(self.filenames[idx])
        image = (image/2**8).astype(np.uint8)
        # normalize
        image = (image - image.mean()) / image.std()
        image = np.expand_dims(image, 0) # for channels
        image = image.astype(np.float32)

        # return torch.from_numpy(image)
        return {'image': torch.from_numpy(image)}
