import os

import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageNet
from PIL import Image

IMAGES_DPATH = '../test/images'
IMAGENET_LOCATION = os.path.expandvars('$HOME/data')

imagenet = ImageNet(IMAGENET_LOCATION,
                    split='val',
                    download=True)

__all__ = [
    'output_to_readable',
    'print_act_pred_labels',
    'init_image_transforms',
    'read_images_as_tensors',
    'save_activations',
    'activations_to_signal',
    'spectrogram_to_signal',
]


def output_to_readable(net_out, imagenet):
    # Highest value corresponds to prediction
    labels_pred_numeric = net_out.argmax(dim=1)
    # Grab classes from ImageNet
    labels_pred = [imagenet.classes[n] for n in labels_pred_numeric]
    return labels_pred


def print_act_pred_labels(labels_act, labels_pred):
    max_len = max([len(l) for l in labels_act])
    n_spaces = max_len - len("Actual")
    print("Index | {}{} | {}".format("Actual",
                                     n_spaces * " ",
                                     "Predicted"))
    for i in range(len(labels_act)):
        n_spaces = max_len - len(labels_act[i])
        print("{:5d} | {}{} | {}".format(i, labels_act[i],
                                         n_spaces * " ",
                                         labels_pred[i]))


def init_image_transforms(size=256):
    if size > 0:
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.transpose(1, 2)
                              if x.shape[1] > x.shape[2] else x),
            transforms.Lambda(lambda x: x.unsqueeze(0)),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.transpose(1, 2)
                              if x.shape[1] > x.shape[2] else x),
            transforms.Lambda(lambda x: x.unsqueeze(0)),
        ])
    return transform


def read_images_as_tensors(image_paths, transform):
    imgs = torch.cat([transform(Image.open(filename))
                      for filename in image_paths])
    return imgs


def save_activations(activations, labels_act,
                     as_pickle=False, as_image=False, dpath=IMAGES_DPATH):

    # Drop them in a sibling directory to test images
    save_dpath = os.path.join(os.path.split(dpath)[0], 'activations')
    labels_save = ["-".join(l.split()) for l in labels_act]

    if as_pickle:

        # Convert activations to numpy arrays for saving as pickles
        activations_np = [a.detach().numpy() for a in activations]
        print('Saving activations as pickles to {}'.format(save_dpath))

        for i in range(len(activations_np)):
            for j in range(len(labels_act)):
                a = activations_np[i][j, ...]
                save_name = os.path.join(save_dpath,
                                         "activations-{}_{}.pkl".format(i + 1, labels_save[j]))  # NOQA
                a.dump(save_name)

    if as_image:
        print('Saving activations as images to {}'.format(save_dpath))
        to_img = transforms.ToPILImage()
        for i in range(len(activations)):
            for j in range(len(labels_act)):
                img = to_img(activations[i][j, ...])
                save_name = os.path.join(save_dpath,
                                         "activations-{}_{}.png".format(i + 1, labels_save[j]))  # NOQA
                img.save(save_name)


def spectrogram_to_signal(spect):
    """ Treating a 2D matrix as a spectrogram, reconstructs the signal
    """
    # Treat columns as time quanta and rows as frequency components
    inv = np.fft.ifft(spect, axis=0).real

    # Scale and shift to [-1, 1]
    inv = (inv - inv.min()) - (inv.max() - inv.min())
    inv = 2 * inv - 1

    # Flatten matrix, treating rows as consecutive samples
    return inv.T.ravel()


def activations_to_signal(activations, combination_method='sum'):
    """ Takes a (D x H x W) activation array and turns it into a signal
        Parameters:
            - activations (np.ndarray): This 3D array is the numpy version of
              the convolutional activations taken from inside a network. Its
              size is (D, H, W)
            - combination_method (str): How the signals extracted from each
              individual activation should be combined. If 'sum', output is
              length HxW. if anything else, length is DxHxW
        Returns:
            - (np.ndarray): shape (L,) where L == HxW if combination_method ==
              'sum' and L == DxHxW if not.
    """
    signal = np.concatenate([spectrogram_to_signal(activations[i, :, :])
                             for i in range(activations.shape[0])])

    if combination_method == 'sum':
        # Sum individuals, scale, and shift
        signal = signal.reshape(activations.shape[0], -1).sum(axis=0)
        signal = (signal - signal.min()) / (signal.max() - signal.min())
        signal = signal * 2 - 1

    return signal
