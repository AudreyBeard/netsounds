import os
import argparse

import numpy as np
from plottingtools import plot

import utils

# TODO:
# [ ] shift reliance from numpy towards torch for speed


def interp(signal, newlen):
    sig = np.interp(np.arange(newlen),
                    np.linspace(0, newlen - 1, signal.shape[0]),
                    signal)
    return sig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--label',
        default='wine_bottle',
        help='labels to use in computation, expect to find videos in data'
             'directory with this in its name',
    )
    parser.add_argument(
        '--combo_method',
        default='concat',
        help='method of combining signals',
    )
    parser.add_argument(
        '--sampling_rate',
        type=int,
        default=44100 // 4,
        help='sampling rate for writing signal',
    )
    parser.add_argument(
        '--tuned',
        action='store_true',
        default=False,
        help='Whether the signals sum or attenuate each other',
    )
    parser.add_argument(
        '--save_signal',
        action='store_true',
        default=False,
        help='whether to save signal',
    )
    return parser.parse_args()


args = parse_args()

dpath = 'test/activations'
filenames = [os.path.join(dpath, l) for l in os.listdir(dpath) if args.label in l]
if len(filenames) == 0:
    raise ValueError('No files found of {} at {}'.format(args.label, dpath))

raw = [utils.activations_to_audio(utils.load_activations(fn), args.combo_method)
       for fn in filenames]

# Order by decreasing length
raw = sorted(raw, key=len)[::-1]

interpped = np.zeros((len(raw), raw[0].size))
interpped[0, :] = raw[0]

for i in range(1, len(raw)):
    interpped[i] = interp(raw[i], interpped.shape[1])

if args.tuned:
    summed = interpped.prod(0)
else:
    summed = interpped.sum(0)
summed = 2 * (summed - summed.min()) / (summed.max() - summed.min()) - 1

# Only save signal if it doesn't exist, since it takes so long
title = 'All {} activations {}'.format(args.combo_method, args.label)
if args.save_signal and not os.path.isfile('graph_{}.png'.format('_'.join(title.split()))):
    print('saving graph...')
    plot.Lines(y=summed,
               title=title).save()

audio_title = 'all-{}{}_{}_{:d}-Hz'.format(
    args.combo_method,
    '_tuned' if args.tuned else '',
    args.label,
    args.sampling_rate
)
utils.save_as_wav(summed, audio_title, args.sampling_rate)
