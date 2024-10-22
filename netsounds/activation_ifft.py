import os
import argparse

import torch
import ipdb  # NOQA
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageNet

import utils
from models.squeezenet import TransparentSqueezeNet


IMAGES_DPATH = 'test/images'
IMAGENET_LOCATION = os.path.expandvars('$HOME/data')
SAMPLING_RATE = 44100


def playground(activations):
    activation_depth = 0
    activation_number = 0

    save_name = 'activation-{}-{}.wav'
    save_path = os.path.join(IMAGES_DPATH, 'sounds', save_name)

    # Grab the first activation and treat it as a spectrogram
    spectrogram = activations[activation_depth][0, activation_number, ...].numpy()  # NOQA
    signal = utils.spectrogram_to_signal(spectrogram)

    utils.save_as_wav(signal, save_path, sampling_rate=SAMPLING_RATE)


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_size',
        type=int,
        default=256,
        help='image size'
    )
    parser.add_argument(
        '--image_name',
        default=None,
        help='a string that appears in the name of the images you want to load'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='whether to drop into an interactive python shell for debugging/testing'
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # Location of test images
    images_fpaths = get_image_paths()
    if args.image_name is not None:
        idx = [i for i in range(len(images_fpaths))
               if args.image_name in images_fpaths[i]][0]
        images_fpaths = [images_fpaths[idx]]

    # Get the actual labels from the filenames
    labels_act = parse_labels_from_paths(images_fpaths)

    # Get ImageNet dataset for interpreting the outputs of the network
    imagenet = ImageNet(IMAGENET_LOCATION,
                        split='val',
                        download=True)

    # For each input image, transform to be same size and orientation
    transform = init_image_transforms(args.image_size)

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

    if args.test:
        import IPython
        IPython.embed()
