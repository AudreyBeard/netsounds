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
