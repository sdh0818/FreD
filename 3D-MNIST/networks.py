import torch.nn as nn

class Conv3DNet(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, im_size):
        super(Conv3DNet, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, im_size)
        num_feat = shape_feat[0] * shape_feat[1] * shape_feat[2] * shape_feat[3]
        self.classifier = nn.Linear(num_feat, num_classes)

    def _get_normlayer(self, shape_feat):
        return nn.InstanceNorm3d(num_features=shape_feat[0], affine=True)

    def _get_activation(self):
        return nn.ReLU(inplace=True)

    def _get_pooling(self):
        return nn.AvgPool3d(kernel_size=(2, 2, 2), stride=2)

    def _make_layers(self, channel, net_width, net_depth, im_size):

        layers = []
        in_channels = channel
        shape_feat = [in_channels, im_size[0], im_size[1], im_size[2]]

        for d in range(net_depth):
            layers += [nn.Conv3d(in_channels, net_width, kernel_size=(3, 3, 3), padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width

            layers += [self._get_normlayer(shape_feat)]
            layers += [self._get_activation()]

            in_channels = net_width

            layers += [self._get_pooling()]
            shape_feat[1] //= 2
            shape_feat[2] //= 2
            shape_feat[3] //= 2

        return nn.Sequential(*layers), shape_feat

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def embed(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

