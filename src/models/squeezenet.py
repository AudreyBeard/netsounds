from torchvision.models.squeezenet import SqueezeNet, Fire
from torch import nn
import torch.utils.model_zoo as model_zoo

# This is where we download the pretrained models from
model_urls = {
    1.0: 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    1.1: 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}


class TransparentSqueezeNet(SqueezeNet):
    """ A more transparent SqueezeNet model
        The exact same architecture as SqueezeNet, but arranged somewhat
        differently from the torchvision implementation. The biggest change is
        that the layers are more separate, allowing us to see intermediate
        activations. This complicates loading a pretrained model, since each
        module needs to be loaded separately. See load_state_dict() for details
        on this.
    """
    def __init__(self, version=1.0, num_classes=1000, pretrained=False):
        super().__init__(version=version, num_classes=num_classes)
        self.version = version

        # Here we copy the architecture of SqueezeNet, but break it apart to see
        # the activations
        # Outputs to each of these will have minimum value of 0, due to ReLU
        if version == 1.0:
            self.features_1 = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),  # 0 - 0
                nn.ReLU(inplace=True),
            )
            self.features_2 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),     # 3 - 1
                Fire(128, 16, 64, 64),    # 4 - 2
                Fire(128, 32, 128, 128),  # 5 - 3
            )
            self.features_3 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),  # 7 - 1
                Fire(256, 48, 192, 192),  # 8 - 2
                Fire(384, 48, 192, 192),  # 9 - 3
                Fire(384, 64, 256, 256),  # 10 4
            )
            self.features_4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),  # 12 - 1
            )
        else:
            raise NotImplementedError("No support for version 1.1 yet")

        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls[version]))

    def forward(self, x):
        """ Same as normal
        """
        x = self.features_1(x)
        x = self.features_2(x)
        x = self.features_3(x)
        x = self.features_4(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)

    def forward_transparent(self, x):
        """ Returns tuple of activations from each module
        """
        x_1 = self.features(x)
        x_2 = self.features(x_1)
        x_3 = self.features(x_2)
        x_4 = self.features(x_3)
        out = self.classifier(x_4)
        return (x_1, x_2, x_3, x_4, out)

    def load_state_dict(self, state_dict):
        """ Replaces superclass's since the arrangement of the modules makes it
            impossible to directly load it.

            Since the loading is very implementation-specific, be very strict
            with the version
        """
        if self.version == 1.0:
            self._load_state_dict_1_0(state_dict)
        else:
            raise NotImplementedError("No support for version 1.1 yet")

    def _load_state_dict_1_0(self, state_dict):
        """ Heavy lifter for loading the state dict into a v1.0 SqueezeNet
        """

        def new_state_dict(prefix_trans, state_dict):
            """ Uses a single prefix translation dictionary to search the
                state_dict for prefix-matched keys, and creates a new dictionary
                with the new prefixes and old suffixes making up the keys, and the
                old data making up the values.
                Parameters:
                    - prefix_trans (dict): dictionary with old state_dict
                      prefixes as keys and new state_dict prefixes as values
                    - state_dict (dict): the model.state_dict(), with layer
                      info as keys and weight tensors as values
                Returns:
                    (dict): with /keys from state_dict that have the same
                    prefix as the keys from prefix_trans, but with those
                    prefixes replaced with values from prefix_trans/ as keys
                    and /values from state_dict corresponding to appropriate
                    keys/ as values
            """
            new_sd = {new_prefix + old_name.split(old_prefix)[1]: tensor
                      for old_name, tensor in state_dict.items()
                      for old_prefix, new_prefix in prefix_trans.items()
                      if old_name.startswith(old_prefix)}
            return new_sd

        # This hardcodey dictionary translates prefixes of
        # SqueezeNet.state_dict() to new prefixes of individual Sequential
        # modules in this class
        prefix_translate = [{'features.0': '0'},
                            {'features.3': '1',
                             'features.4': '2',
                             'features.5': '3'},
                            {'features.7': '1',
                             'features.8': '2',
                             'features.9': '3',
                             'features.10': '4'},
                            {'features.12': '1'},
                            {'classifier.1': '1'}]

        state_dicts = [new_state_dict(item, state_dict)
                       for item in prefix_translate]

        self.features_1.load_state_dict(state_dicts[0])
        self.features_2.load_state_dict(state_dicts[1])
        self.features_3.load_state_dict(state_dicts[2])
        self.features_4.load_state_dict(state_dicts[3])
        self.classifier.load_state_dict(state_dicts[4])
