import os
import argparse

import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageNet
import ipdb  # NOQA

from squeezenet import TransparentSqueezeNet


IMAGES_DPATH = '../test/images'
IMAGENET_LOCATION = os.path.expandvars('$HOME/data')


def to_readable(net_out, imagenet):
    # Highest value corresponds to prediction
    labels_pred_numeric = net_out.argmax(dim=1)
    # Grab classes from ImageNet
    labels_pred = [imagenet.classes[n] for n in labels_pred_numeric]
    return labels_pred


def print_act_predict(labels_act, labels_pred):
    max_len = max([len(l) for l in labels_act])
    n_spaces = max_len - len("Actual")
    print("Index | {}{} | {}".format("Actual", n_spaces * " ", "Predicted"))
    for i in range(len(labels_act)):
        n_spaces = max_len - len(labels_act[i])
        #print("Actual   : {}".format(labels_act[i]))
        #print("Predicted: {}".format(labels_pred[i]))
        print("{:5d} | {}{} | {}".format(i, labels_act[i], n_spaces * " ", labels_pred[i]))


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
                                         "activations-{}_{}.pkl".format(i + 1, labels_save[j]))
                a.dump(save_name)

    if as_image:
        print('Saving activations as images to {}'.format(save_dpath))
        to_img = transforms.ToPILImage()
        for i in range(len(activations)):
            for j in range(len(labels_act)):
                img = to_img(activations[i][j, ...])
                save_name = os.path.join(save_dpath,
                                         "activations-{}_{}.png".format(i + 1, labels_save[j]))
                img.save(save_name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pickles',
        action='store_true',
        default=False,
        help='whether to save activations as pickles'
    )
    parser.add_argument(
        '--images',
        action='store_true',
        default=False,
        help='whether to save activations as images'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=256,
        help='image size (small edge)'
    )
    parser.add_argument(
        '--image_name',
        default='wine_bottle',
        help='image name'
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # Location of test images
    images_fpaths = get_image_paths()
    if args.image_name is not None:
        idx = [i for i in range(len(images_fpaths)) if args.image_name in images_fpaths[i]][0]
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

    # Pass images through model
    output_opaque = model(imgs)

    # Predicted label
    labels_pred = to_readable(output_opaque, imagenet)

    #with ipdb.launch_ipdb_on_exception():
    with torch.no_grad():
        transparent_out = model.forward_transparent(imgs)

    # Hardcoded based on output of forward_transparent()
    activations = [transparent_out[i] for i in range(4)]
    output_transparent = transparent_out[-1]

    # Predicted label
    labels_pred_trans = to_readable(output_transparent, imagenet)

    # Prediction information:
    print("Opaque:")
    print_act_predict(labels_act, labels_pred_trans)
    print("\nTransparent:")
    print_act_predict(labels_act, labels_pred_trans)

    # Save the activations as pickles of numpy arrays
    save_activations(activations, labels_act,
                     as_pickle=args.pickles,
                     as_image=args.images)
