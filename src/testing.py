import os

import numpy as np
from plottingtools import plot
import ubelt as ub

import utils


def interp(signal, newlen):
    sig = np.interp(np.arange(newlen),
                    np.linspace(0, newlen - 1, signal.shape[0]),
                    signal)
    return sig


label = ub.argval('label', default='wine-bottle')
method = ub.argval('method', default='concat')
sampling_rate = int(ub.argval('sampling_rate', default=44100//4))
tuned = ub.argflag('--tuned')
show = ub.argflag('--show')
dpath = 'test/activations'
filenames = [os.path.join(dpath, l) for l in os.listdir(dpath) if label in l]
if len(filenames) == 0:
    raise ValueError('No files found of {} at {}'.format(label, dpath))

raw = [utils.activations_to_audio(utils.load_activations(fn), method)
       for fn in filenames]

# Order by decreasing length
raw = sorted(raw, key=len)[::-1]

interpped = np.zeros((len(raw), raw[0].size))
interpped[0, :] = raw[0]

for i in range(1, len(raw)):
    interpped[i] = interp(raw[i], interpped.shape[1])

if tuned:
    summed = interpped.prod(0)
else:
    summed = interpped.sum(0)
summed = 2 * (summed - summed.min()) / (summed.max() - summed.min()) - 1

title = 'All {} activations {}'.format(method, label)
if show and not os.path.isfile('graph_{}.png'.format('_'.join(title.split()))):
    print('saving graph...')
    plot.Lines(y=summed,
               title=title).save()
audio_title = 'all-{}_{}_{}_{}-Hz'.format(
    method,
    'tuned' if tuned else '',
    label,
    sampling_rate
)
utils.save_as_wav(summed, audio_title, sampling_rate)
