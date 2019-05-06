# netsounds
Sonification of convolutional neural networks

## Functionality
Currently, we've got a modified pretrained SqueezeNet v1.0 running some home
images, and dumping the activations at certain layers out as numpy pickles.

To do this, run `python3 test_squeezenet.py` from `/src/models/`

## TODO
- [ ] Dump images in `test_squeezenet.py`
- [ ] Explore the activations at different layers

## Notes
- activations in this repo are generated from images with small edge length of
  256:
    - This means that the activations are very small
    - To get higher-fidelity activations, we must use larger images, but we
      quickly run out of memory (and storage on GitHub)
- I've tried putting larger activations in, but my machine (8GB RAM) runs out
  of memory on anything larger than 1024
- I put the activations for `image_size=1024` in
  [here](https://drive.google.com/open?id=115nEdJh2cO4Ei2a2y0DyLTczBpaZs7Tr)
