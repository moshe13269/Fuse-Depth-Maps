import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class _DoubleConvolution(nn.Module):
    def __init__(self, in_channels, middle_channel, out_channels, p=0, drop_p=0):
        super(_DoubleConvolution, self).__init__()
        layers = [
            nn.Conv2d(in_channels, middle_channel, kernel_size=3, padding=p),
            nn.BatchNorm2d(middle_channel),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop_p),
            nn.Conv2d(middle_channel, out_channels, kernel_size=3, padding=p),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop_p)
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(_AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return g*(1-psi), x*psi


class _DenseLayer(nn.Module):
    def __init__(self, n_channels, growth_rate=16, drop_p=0):
        """
        DenseNet Layer
        Returns concatenation of input and output along feature map
        dimension `1`.

        BN -> ReLU -> 3x3 Conv -> 2D Dropout (p)

        Parameters
        ----------
        n_channels : int. number of input channels.
        growth_rate : int. growth rate `k`, number of feature maps to add
            to the input before concatenation and output.
        """

        super(_DenseLayer, self).__init__()

        self.n_channels = n_channels
        self.growth_rate = growth_rate
        self.drop_p = drop_p

        self.bn = nn.BatchNorm2d(self.n_channels)
        self.conv = nn.Conv2d(self.n_channels, self.growth_rate, kernel_size=3, padding=1)  # keep shape , change number of channels to grow_rate
        self.do = nn.Dropout2d(p=drop_p)

    def forward(self, x):
        out0 = F.relu(self.bn(x))
        out1 = self.conv(out0)
        out2 = self.do(out1)
        concat = torch.cat([x, out2], 1)
        return concat


class _TransitionDown(nn.Module):
    def __init__(self, n_channels_in, n_channels_out=None, drop_p=0, pool=True):
        """
        Transition Down module.
        Returns downsampled image, preserving the number of feature maps by
        default.

        BN -> ReLU -> 1x1 Conv -> 2D Dropout (p) -> Max Pooling

        Parameters
        ----------
        n_channels_in : int. number of input channels
        n_channels_out : int, optional. number of output channels.
            preserves input by default.
        """
        super(_TransitionDown, self).__init__()

        self.drop_p = drop_p

        self.n_channels_in = n_channels_in
        if n_channels_out is not None:
            self.n_channels_out = n_channels_out
        else:
            self.n_channels_out = self.n_channels_in

        self.bn = nn.BatchNorm2d(self.n_channels_in)
        self.conv = nn.Conv2d(self.n_channels_in, self.n_channels_out, kernel_size=1, padding=0)
        self.do = nn.Dropout2d(p=drop_p)

        if pool is True:
            self.pool = nn.MaxPool2d((2, 2), stride=2)
        else:
            self.pool = nn.Identity()

    def forward(self, x):
        out0 = F.relu(self.bn(x))
        out1 = self.conv(out0)
        out2 = self.do(out1)
        pooled = self.pool(out2)
        return pooled


class _TransitionUp(nn.Module):
    def __init__(self, n_channels_in, n_channels_out=None):
        """
        FC-DenseNet Transition Up module
        Returns upsampled image by transposed convolution.

        3 x 3 Transposed Conv stride = 2

        Parameters
        ----------
        n_channels_in : int. number of input channels
        n_channels_out : int, optional. number of output channels.
            preserves input by default.
        """
        super(_TransitionUp, self).__init__()

        self.n_channels_in = n_channels_in
        if n_channels_out is not None:
            self.n_channels_out = n_channels_out
        else:
            self.n_channels_out = self.n_channels_in

        # pad input and output by `1` to maintain (x,y) size
        self.transconv = nn.ConvTranspose2d(self.n_channels_in, self.n_channels_out, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.transconv = nn.ConvTranspose2d(self.n_channels_in, self.n_channels_out, kernel_size=2, stride=2)

    def forward(self, x):
        up = self.transconv(x)
        return up


class _DenseBlock(nn.Module):
    def __init__(self, n_layers, n_channels, growth_rate=16, drop_p=0, keep_input=True):
        """
        Builds a DenseBlock from DenseLayers.

        Parameters
        ----------
        n_layers : int. number of DenseLayers in the block.
        n_channels : int. number of input channels.
        growth_rate : int. growth rate `k`, number of feature maps to add
            to the input before concatenation and output.
        keep_input : boolean. concatenate the input to the newly added
            feature maps from this DenseBlock. input concatenation is omitted
            for DenseBlocks in the upsampling path of FC-DenseNet103.
        """
        super(_DenseBlock, self).__init__()

        self.n_layers = n_layers
        self.n_channels = n_channels
        self.growth_rate = growth_rate
        self.keep_input = keep_input
        self.drop_p = drop_p

        if self.keep_input:
            self.n_channels_out = n_channels + self.growth_rate*self.n_layers
        else:
            self.n_channels_out = self.growth_rate*self.n_layers

        self.block = self._build_block()

    def forward(self, x):
        out = self.block(x)
        if not self.keep_input:
            out = out[:, self.n_channels:, ...]  # omit input feature maps
        return out

    def _build_block(self):
        n_channels = self.n_channels
        layers = []

        for i in range(self.n_layers):
            ll = _DenseLayer(n_channels, self.growth_rate, self.drop_p)
            layers.append(ll)
            n_channels += self.growth_rate

        stack = nn.Sequential(*layers)
        return stack


class DUNet(nn.Module):
    def __init__(self, n_channels_in=1,
                 growth_rate=4,
                 n_channels_first=8,
                 n_layers_down=[4, 5, 7],
                 n_layers_up=[7, 5, 4],
                 drop_p=0.2,
                 n_bottleneck_layers=9,
                 verbose=False):

        super(DUNet, self).__init__()

        self.growth_rate = growth_rate
        self.n_channels_in = n_channels_in
        self.n_channels_first = n_channels_first
        self.n_layers_down = n_layers_down
        self.n_layers_up = n_layers_up
        self.path_len = len(n_layers_down)
        self.drop_p = drop_p
        self.n_bottleneck_layers = n_bottleneck_layers
        self.verbose = verbose

        if len(n_layers_down) != len(n_layers_up):
            raise ValueError('`n_layers_down` must be as the length of `n_layers_up`')
        else:
            pass

        self.conv0_in1 = nn.Conv2d(self.n_channels_in, self.n_channels_first, kernel_size=3, stride=1, padding=1)
        self.conv0_in2 = nn.Conv2d(self.n_channels_in, self.n_channels_first, kernel_size=3, stride=1, padding=1)

        self.denseB_in1 = _DenseBlock(n_layers=3, n_channels=self.n_channels_first, growth_rate=4, drop_p=self.drop_p)
        self.denseB_in2 = _DenseBlock(n_layers=3, n_channels=self.n_channels_first, growth_rate=4, drop_p=self.drop_p)

        self.att_in = _AttentionBlock(self.denseB_in1.n_channels_out,
                                      self.denseB_in2.n_channels_out,
                                      self.denseB_in2.n_channels_out)

        # Downsampling path
        down_channels = self.denseB_in1.n_channels_out + self.denseB_in1.n_channels_out # self.n_channels_first
        skip_channels = []
        for i in range(self.path_len):
            setattr(self, 'down_dblock' + str(i), _DenseBlock(n_layers=self.n_layers_down[i],
                                                              n_channels=down_channels,
                                                              growth_rate=self.growth_rate,
                                                              drop_p=self.drop_p))
            down_channels = getattr(self, 'down_dblock' + str(i)).n_channels_out

            setattr(self, 'td' + str(i), _TransitionDown(n_channels_in=down_channels, drop_p=self.drop_p))

            skip_channels.append(down_channels)

        # Bottleneck
        self.bottleneck = _DenseBlock(n_layers=self.n_bottleneck_layers,
                                      n_channels=getattr(self, 'down_dblock'+str(self.path_len-1)).n_channels_out,
                                      growth_rate=self.growth_rate,
                                      drop_p=self.drop_p,
                                      keep_input=False)

        # Upsampling path
        up_channels = self.bottleneck.n_channels_out
        for i in range(self.path_len):
            keep = False
            setattr(self, 'tu' + str(i), _TransitionUp(n_channels_in=up_channels))

            s_chan = skip_channels[-(i+1)]
            udb_channels = up_channels + s_chan

            setattr(self, 'att_up' + str(i), _AttentionBlock(up_channels, s_chan, s_chan))

            setattr(self, 'tu_dconv' + str(i), _TransitionDown(n_channels_in=udb_channels,
                                                               drop_p=self.drop_p, pool=False))

            setattr(self, 'up_dblock' + str(i), _DenseBlock(n_layers=self.n_layers_up[i],
                                                            n_channels=udb_channels,
                                                            growth_rate=self.growth_rate,
                                                            drop_p=self.drop_p,
                                                            keep_input=keep))

            up_channels = getattr(self, 'up_dblock' + str(i)).n_channels_out

        self.conv1 = nn.Conv2d(getattr(self, 'up_dblock' + str(self.path_len-1)).n_channels_out, 1, kernel_size=1)

        initialize_weights(self)

    def forward(self, x_in1, x_in2):
        in_conv1 = self.conv0_in1(x_in1)
        in_conv2 = self.conv0_in2(x_in2)

        d_out1 = self.denseB_in1(in_conv1)
        d_out2 = self.denseB_in2(in_conv2)

        d_out1, d_out2 = self.att_in(d_out1, d_out2)

        out = torch.cat((d_out1, d_out2), 1)

        # Downsampling path
        self.dblock_outs = []
        for i in range(self.path_len):

            dblock = getattr(self, 'down_dblock' + str(i))
            td = getattr(self, 'td' + str(i))

            db_x = dblock(out)
            self.dblock_outs.append(db_x)

            out = td(db_x)
            if self.verbose:
                print('m: ', out.size(1))

        # Bottleneck
        bneck = self.bottleneck(out)
        if self.verbose:
            print('bottleneck m: ', bneck.size(1) + out.size(1))

        # Upsampling path
        out = bneck
        for i in range(self.path_len):

            tu = getattr(self, 'tu' + str(i))
            tu_dconv = getattr(self, 'tu_dconv' + str(i))
            ublock = getattr(self, 'up_dblock' + str(i))
            att_up = getattr(self, 'att_up' + str(i))
            skip = self.dblock_outs[-(i+1)]

            up = tu(out)

            up, skip = att_up(up, skip)
            cat = tu_dconv(torch.cat([up, skip], 1))

            out = ublock(cat)

            if self.verbose:
                print('Skip: ', skip.size())
                print('Up: ', up.size())
                print('Cat: ', cat.size())
                print('Out: ', out.size())
                print('m : ', cat.size(1) + out.size(1))

        final = self.conv1(out)
        if self.verbose:
            print('Result: ', final.size())

        return final

    # @staticmethod
    # def center_crop(x, y):
    #     if x.shape[0] != y.shape[0]:
    #         raise ValueError(f'x and y inputs contain a different number of samples')
    #
    #     def crop_tensor(in_tensor, target_height, target_width):
    #         current_height = in_tensor.size(2)
    #         current_width = in_tensor.size(3)
    #         min_w = (current_width - target_width) // 2
    #         min_h = (current_height - target_height) // 2
    #         return in_tensor[:, :, min_h:(min_h + target_height), min_w:(min_w + target_width)]
    #
    #     height = min(x.size(2), y.size(2))
    #     width = min(x.size(3), y.size(3))
    #
    #     cropped_x = crop_tensor(x, height, width)
    #     cropped_y = crop_tensor(y, height, width)
    #
    #     # res = torch.cat([cropped_x, cropped_y], dim=1)
    #
    #     return cropped_x, cropped_y


# m = DUNet(1)
# torch_total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
# print('DUNet Total Params:', torch_total_params)

# im1 = torch.rand(1, 1, 1024, 512)
# im2 = torch.rand(1, 1, 1024, 512)
# m_out = m(im1, im2)


