import os

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

IMAGES_DPATH = '../test/images'
IMAGENET_LOCATION = os.path.expandvars('$HOME/data')


__all__ = [
    'output_to_readable',
    'print_act_pred_labels',
    'init_image_transforms',
    'read_images_as_tensors',
    'save_activations',
    'activations_to_audio',
    'spectrogram_to_signal',
    'save_as_wav',
]


def get_images_from_name_fragment(name_fragment=None):
    # Location of test images
    images_fpaths = get_image_paths()
    if name_fragment is not None:
        idx = [
            i
            for i in range(len(images_fpaths))
            if name_fragment in images_fpaths[i]
        ]
        images_fpaths = [images_fpaths[i] for i in idx]
    return images_fpaths


def get_image_paths(dpath=IMAGES_DPATH):
    dpath = os.path.realpath(dpath)
    images_fpaths = [os.path.join(dpath, filename)
                     for filename in os.listdir(dpath)]
    return images_fpaths


def init_imagenet(dpath=IMAGENET_LOCATION):
    """
        Initializes Imagenet dataset
    """
    from torchvision.datasets import ImageNet
    imagenet = ImageNet(IMAGENET_LOCATION,
                        split='val',
                        download=True)
    return imagenet


def output_to_readable(net_out, imagenet=None):
    """ Converts network output to plaintext labels
    """
    if imagenet is None:
        imagenet = init_imagenet(IMAGENET_LOCATION)

    # Highest value corresponds to prediction
    labels_pred_numeric = net_out.argmax(dim=1)
    # Grab classes from ImageNet
    labels_pred = [imagenet.classes[n] for n in labels_pred_numeric]
    return labels_pred


def print_act_pred_labels(labels_act, labels_pred):
    """ Prints actual and predicted labels nicely
    """
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
    """ Initialize transforms of data
    """
    from torchvision import transforms
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
    """ Read in images as tensors
    """
    from torch import cat
    imgs = cat([transform(Image.open(filename))
                for filename in image_paths])
    return imgs


def cache_activations(activations, labels, dpath=None):
    """ Save activations as torch files (.t7)
    """
    # Drop them in a sibling directory to test images
    if dpath is None:
        dpath = os.path.join(os.path.split(IMAGES_DPATH)[0],
                             'activations')

    # Replace spaces with hypens
    labels_nospace = ["-".join(l.split()) for l in labels]

    for i in range(activations.shape[0]):
        for j in range(len(labels_nospace)):
            save_name = os.path.join(
                dpath,
                "activations-{}_{}.t7".format(
                    i + 1,
                    labels_nospace[j]
                )
            )
            torch.save(activations[i, j, :], save_name)
    return


def image_activations(activations, labels, dpath=None):
    # Drop them in a sibling directory to test images
    if dpath is None:
        dpath = os.path.join(os.path.split(IMAGES_DPATH)[0],
                             'activations')
    # Replace spaces with hypens
    labels_nospace = ["-".join(l.split()) for l in labels]
    to_img = transforms.ToPILImage()
    for i in range(activations.shape[0]):
        for j in range(len(labels_nospace)):
            img = to_img(activations[i, j, :])
            save_name = os.path.join(
                dpath,
                "activations-{}_{}.png".format(
                    i + 1,
                    labels_nospace[j]
                )
            )
            img.save(save_name)
    return


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


def load_activations(fpath):
    return np.load(fpath, allow_pickle=True)


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


def activations_to_audio(activations, combination_method='sum'):
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
    # signal = np.concatenate([spectrogram_to_signal(activations[i, :, :])
    #                          for i in range(activations.shape[0])])
    # This is probably faster
    signal = np.fft.ifft(activations, axis=-2).real
    signal = signal.transpose(0, 2, 1).reshape(signal.shape[0], -1)

    # Standardize now for a few reasons:
    # 1. This weights each filter equally, as opposed to high-activation
    #    filters overpowering low-activation filters
    # 2. This makes for a smoother signal (due to 1) and is therefore less
    #    harsh
    # 3. Simpler code

    sigmax = signal.max(axis=1)[:, None]
    sigmin = signal.min(axis=1)[:, None]
    # This prevents divide-by-zero errors, which may pop up when an entire
    # filter fails to activate
    div = np.where(sigmax - sigmin, sigmax - sigmin, 1)
    signal = 2 * (signal - sigmin) / div - 1

    if combination_method == 'sum':
        # Sum filters
        signal = signal.sum(axis=0)

        sigmax = signal.max()
        sigmin = signal.min()
        # This prevents divide-by-zero errors, which may pop up when an entire
        # filter fails to activate
        div = np.where(sigmax - sigmin, sigmax - sigmin, 1)
        signal = 2 * (signal - sigmin) / div - 1

    else:
        # Concatenate filters
        signal = signal.ravel()

    return signal


def save_as_wav(signal, save_name, sampling_rate=44100):
    from scipy.io.wavfile import write
    if not save_name.endswith('.wav'):
        save_name += '.wav'
    write(save_name, sampling_rate, signal)
