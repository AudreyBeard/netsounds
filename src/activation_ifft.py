import os

import torch
import numpy as np
import ipdb  # NOQA
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageNet

from models.squeezenet import TransparentSqueezeNet


IMAGES_DPATH = 'test/images'
IMAGENET_LOCATION = os.path.expandvars('$HOME/data')
SAMPLING_RATE = 44100


def playground(activations):
    from scipy.io.wavfile import write
    activation_depth = 0
    activation_number = 0

    save_name = 'activation-{}-{}.wav'
    save_path = os.path.join(IMAGES_DPATH, 'sounds', save_name)

    # Grab the first activation and treat it as a spectrogram
    spectrogram = activations[activation_depth][0, activation_number, ...].numpy()
    inversed = np.fft.ifft(spectrogram, axis=0).real
    inversed = inversed - inversed.min()
    inversed = inversed / inversed.max()
    inversed = 2 * inversed - 1
    inversed = inversed.T
    signal = inversed.ravel()

    #signal = np.int16(signal * 32767)
    write(save_path, SAMPLING_RATE, signal)


def get_image_paths(dpath=IMAGES_DPATH):
    dpath = os.path.realpath(dpath)
    images_fpaths = [os.path.join(dpath, filename)
                     for filename in os.listdir(dpath)]
    return images_fpaths


def parse_labels_from_paths(image_paths):
    labels = [' '.join((os.path.splitext(os.path.split(fp)[-1])[0]).split('_'))
              for fp in image_paths]
    return labels


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
                      for filename in images_fpaths])
    return imgs


def to_readable(net_out, imagenet):
    # Highest value corresponds to prediction
    labels_pred_numeric = net_out.argmax(dim=1)
    # Grab classes from ImageNet
    labels_pred = [imagenet.classes[n] for n in labels_pred_numeric]
    return labels_pred


if __name__ == "__main__":
    import ubelt as ub

    # Get input arguments
    save_activation_pickles = ub.argflag('--save')
    save_activation_images = ub.argflag('--save_imgs')
    image_size = int(ub.argval('image_size', default=256))
    image_name = ub.argval('image_name', default=None)

    # Location of test images
    images_fpaths = get_image_paths()
    if image_name is not None:
        idx = [i for i in range(len(images_fpaths)) if image_name in images_fpaths[i]][0]
        images_fpaths = [images_fpaths[idx]]

    # Get the actual labels from the filenames
    labels_act = parse_labels_from_paths(images_fpaths)

    # Get ImageNet dataset for interpreting the outputs of the network
    imagenet = ImageNet(IMAGENET_LOCATION,
                        split='val',
                        download=True)

    # For each input image, transform to be same size and orientation
    transform = init_image_transforms(image_size)

    # Put all images into a big tensor for putting into model
    imgs = read_images_as_tensors(images_fpaths, transform)

    # Instantiate model
    model = TransparentSqueezeNet(pretrained=True)
    with torch.no_grad():
        transparent_out = model.forward_transparent(imgs)

    # Hardcoded based on output of forward_transparent()
    activations = [transparent_out[i] for i in range(4)]
    output_transparent = transparent_out[-1]

    # Predicted label
    labels_pred_trans = to_readable(output_transparent, imagenet)

    import IPython
    IPython.embed()
