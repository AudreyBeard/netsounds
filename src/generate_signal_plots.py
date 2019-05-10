import os

import utils
from plottingtools import plot

activation_dpath = './test/activations'
sounds_dpath = './test/sounds'
sampling_rate = 44100

activation_files = os.listdir(activation_dpath)
activation_files = [os.path.join(activation_dpath, fn)
                    for fn in activation_files if fn.endswith('.pkl')]

labels = [os.path.splitext(os.path.split(fn)[1])[0]
          for fn in activation_files]

for i in range(len(activation_files)):
    activations = utils.load_activations(activation_files[i])
    for method in ['sum', 'concat']:
        signal = utils.activations_to_audio(activations, method)

        save_name = os.path.join(sounds_dpath,
                                 '{}_{}.png'.format(labels[i], method))
        title = '{}_{}'.format(labels[i], method)
        plot.Lines(y=signal,
                   title=title).save()
        print(title, flush=True)
    plot.close_all()

print('Done')
