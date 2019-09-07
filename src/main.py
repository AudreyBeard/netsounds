import os

import argparse
import torch

from models.squeezenet import TransparentSqueezeNet
import utils

# Higher verbosity means more command-line output
VERBOSITY = 1

# Download models to ~/.cache (implicitly)
# Download data to ~/data (explicitly)
data_dir = os.path.expandvars('$HOME/data')
if VERBOSITY > 0:
    print("Data location: {}".format(data_dir))

#model = TransparentSqueezeNet(pretrained=True)
#dataset = ImageNet(root=data_dir, split='val', download=True)


# TODO test this - it's just a copy from models/test_squeezenet.py
def parse_args():
    parser = argparse.ArgumentParser()
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    image_fpaths = utils.get_images_from_name_fragment(args.image_name)
    imagenet = utils.init_imagenet()
    transform = utils.init_image_transforms(args.image_size)
    imgs = utils.read_images_as_tensors(image_fpaths, transform)
    model = TransparentSqueezeNet(pretrained=True)

    with torch.no_grad():
        transparent_out = model.forward_transparent(imgs)

    # Hardcoded based on output of forward_transparent()
    activations = [transparent_out[i] for i in range(4)]
    output = transparent_out[-1]

    labels_pred = utils.output_to_readable(output, imagenet)
    utils.cache_activations(activations, labels_pred)
