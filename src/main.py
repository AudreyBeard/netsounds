import os

# 2019-05-05: Removed from pip-installable version for some reason???
#from torchvision.datasets import ImageNet

from models.squeezenet import TransparentSqueezeNet

# Higher verbosity means more command-line output
VERBOSITY = 1

# Download models to ~/.cache (implicitly)
# Download data to ~/data (explicitly)
data_dir = os.path.expandvars('$HOME/data')
if VERBOSITY > 0:
    print("Data location: {}".format(data_dir))

model = TransparentSqueezeNet(pretrained=True)
#dataset = ImageNet(root=data_dir, split='val', download=True)


# TODO test this
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
