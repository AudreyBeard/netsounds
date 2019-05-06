import os
import argparse

import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageNet
import ipdb  # NOQA

from squeezenet import TransparentSqueezeNet

save = True
parser = argparse.ArgumentParser()
parser.add_argument('--save',
                    dest='save_activations',
                    nargs='?',
                    default=False,
                    help='Use this flag if you want to save the activations as pickles')
parser.add_argument('--save_images',
                    dest='save_images',
                    nargs='?',
                    default=False,
                    help='Use this flag if you want to save the activations as images')


def to_readable(net_out, imagenet):
    # Highest value corresponds to prediction
    labels_pred_numeric = output_opaque.argmax(dim=1)
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


if __name__ == "__main__":
    args = parser.parse_args()

    # Location of test images
    images_dpath = os.path.realpath('../test/images')
    images_fpaths = [os.path.join(images_dpath, filename)
                     for filename in os.listdir(images_dpath)]

    # Get the actual labels from the filenames
    labels_act = [' '.join((os.path.splitext(os.path.split(fp)[-1])[0]).split('_'))
                  for fp in images_fpaths]

    # Get ImageNet dataset for interpreting the outputs of the network
    imagenet_location = os.path.expandvars('$HOME/data')
    imagenet = ImageNet(imagenet_location,
                        split='val',
                        download=True)

    # For each input image, transform to be same size and orientation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.transpose(1, 2)
                          if x.shape[1] > x.shape[2] else x),
        transforms.Lambda(lambda x: x.unsqueeze(0)),
    ])

    # Put all images into a big tensor for putting into model
    imgs = torch.cat([transform(Image.open(filename))
                      for filename in images_fpaths])

    # Instantiate model
    model = TransparentSqueezeNet(pretrained=True)

    # Pass images through model
    output_opaque = model(imgs)

    # Predicted label
    labels_pred = to_readable(output_opaque, imagenet)

    #with ipdb.launch_ipdb_on_exception():
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
    if args.save_activations:
        save_dpath = os.path.join(os.path.split(images_dpath)[0], 'activations')
        activations = [a.detach().numpy() for a in activations]
        for i in range(len(activations)):
            for j in range(len(labels_act)):
                label_save = "-".join(labels_act[j].split())
                a = activations[i][j, ...]
                save_name = os.path.join(save_dpath,
                                         "activations-{}_{}.pkl".format(i + 1, label_save))
                a.dump(save_name)

    if args.save_images:
        raise NotImplementedError("this is a wip")
        # TODO wip
        save_dpath = os.path.join(os.path.split(images_dpath)[0], 'activations')
        to_img = transforms.ToPILImage()
        for i in range(len(activations)):
            for j in range(len(labels_act)):
                # Since the activations are N x F x cH x cW, we want either:
                # N, F, NxF images saved for each activation level. I need to
                # figure out the logic here, and save them for Ezra to get some
                # insight about
                activations = [to_img(a.detach()) for a in activations]
                label_save = "-".join(labels_act[j].split())
                a = activations[i][j, ...]
                save_name = os.path.join(save_dpath,
                                         "activations-{}_{}.pkl".format(i + 1, label_save))
                a.dump(save_name)
