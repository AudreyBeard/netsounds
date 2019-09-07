# netsounds
Sonification of convolutional neural networks


## Usage:
To get started, simply call `python main.py --dpath=directory/containing/photos --image_name=fragment_of_an_image_name`
This will:
1. Grab an image
2. Pass the image through a network (SqueezeNet), grabbing:
    - Intermediate activations from four filter banks
    - Output prediction
3. Save activations in `./test/activations` with the naming convention:
   `activations-X_LABEL.t7`, with `X` being the bank index, and `LABEL` being
   the prediction
4. Save audio signals generated from IFFT of activations, treating them like
   spectrograms


## TODO
- [x] `torch.ifft()` instead of `np.fft.ifft()`
  - This will make the process faster, since we're not switching between
    `torch.tensor`s and `np.ndarray`s
- [ ] Dump images in `test_squeezenet.py`
- [ ] Explore the activations at different layers
    - Made sounds for each activation using two generation schemes:
        - Concatenating each filter at a level
        - Summing all filters at a level
- [ ] Put as much of this on a GPU as possible:
  - Use PyTorch as much as possible
  - Make sure we can still use CPU (in case of use on Pi, for instance)


## Functionality
### Model
Currently, we've got a modified pretrained SqueezeNet v1.0 running some home
images, and dumping the activations at certain layers out as numpy pickles.

To do this, run `python3 test_squeezenet.py` from `/src/models/`

### Sound
`utils.py` has a function that writes an activation to a sound. It's
pretty rad. Right now it wrote out the audio for the first activation from a
1024 px image of `mixing_bowl.jpg`


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
- for the tensor implementation of signal derivation, the IFFT is taken on the
  last dimension of the activation tensor, because it's faster than doing it on
  the second-to-last one, followed by a permutation.
