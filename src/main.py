import argparse
import torch

from models.squeezenet import TransparentSqueezeNet
import utils


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
        help='Image name, or fragment of an image name. If multiple matching '
             'files exist, will use the first (alphanumerically sorted)'
    )
    parser.add_argument(
        '--dpath',
        default=utils.IMAGES_DPATH,
        help='location to search for images'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Drop into debugger?'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Initialize Imagenet dataset, image transform, and prediction model
    imagenet = utils.init_imagenet()
    transform = utils.init_image_transforms(args.image_size)
    model = TransparentSqueezeNet(pretrained=True)

    # Get an image filepath that contains user-specified image name fragment
    image_fpaths = utils.get_images_from_name_fragment(args.image_name, args.dpath)[0]

    # Read image as a tensor for passing through the network
    imgs = utils.read_images_as_tensors(image_fpaths, transform)

    # No gradient computations need to be done, since we're not training
    with torch.no_grad():

        # Pass data through the model to get activations and output
        transparent_out = model.forward_transparent(imgs)
        activations = transparent_out[:-1]
        output = transparent_out[-1]

    labels_pred = utils.output_to_readable(output, imagenet)
    utils.cache_activations(activations, labels_pred)

    if args.debug:
        import ipdb
        ipdb.set_trace()

    utils.save_wavs(activations, labels_pred)
