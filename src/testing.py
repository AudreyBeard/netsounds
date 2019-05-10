import os

import numpy as np
from plottingtools import plot

import utils


def interp(signal, newlen):
    sig = np.interp(np.arange(newlen),
                    np.linspace(0, newlen - 1, signal.shape[0]),
                    signal)
    return sig


label = 'mixing-bowl'
dpath = 'test/activations'
method = 'sum'
sampling_rate = 44100//4
filenames = [os.path.join(dpath, l) for l in os.listdir(dpath) if label in l]

raw = [utils.activations_to_audio(utils.load_activations(fn), method)
       for fn in filenames]

# Order by decreasing length
raw = sorted(raw, key=len)[::-1]

interpped = np.zeros((len(raw), raw[0].size))
interpped[0, :] = raw[0]

for i in range(1, len(raw)):
    interpped[i] = interp(raw[i], interpped.shape[1])

summed = interpped.sum(0)
summed = 2 * (summed - summed.min()) / (summed.max() - summed.min()) - 1

title = 'All {} activations {}'.format(method, label)
if not os.path.isfile('graph_{}.png'.format('_'.join(title.split()))):
    print('saving graph...')
    plot.Lines(y=summed,
               title=title).save()

utils.save_as_wav(summed, 'all-{}_{}'.format(method, label), sampling_rate)
