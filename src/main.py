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
    parser.add_argument(
        '--combo_method',
        default='concat',
        choices=['concat', 'sum', 'none'],
        help='Method by which we combine activation signals'
    )
    parser.add_argument(
        '--sampling_rate',
        default=441000 // 4,
        type=int,
        help='sampling rate for writing the file'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.debug:
        import ipdb
        ipdb.set_trace()

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
    save_names_activations = utils.cache_activations(activations, labels_pred)

    save_names_sounds = utils.save_wavs(
        activations, labels_pred[0],
        combination_method=args.combo_method,
        sampling_rate=args.sampling_rate
    )

    print("Activations are at:")
    for name in save_names_activations:
        print("  {}".format(name))
    print("Sounds are at:")
    if isinstance(save_names_sounds, str):
        print("  {}".format(save_names_sounds))
    else:
        for name in save_names_sounds:
            print("  {}".format(name))
