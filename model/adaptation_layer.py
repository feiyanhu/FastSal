import torch as t
import torch.nn as nn
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.mobilenet import ConvBNReLU
from collections import OrderedDict

from .utils import register_layers


class CNN_constructer(nn.Sequential):
    def __init__(self, networklist, upsample_mode='bilinear', output_activation='relu',
                 kernel_size=3, padding_n=1, depth=False, encoder_hook_index=None, decoder_hook_index=None):
        d = OrderedDict()
        # next_feature = in_channels
        prev_conv = networklist[0]
        networklist = networklist[1:]
        for layer_idx, layer_conv in enumerate(networklist):
            if 'M' in str(layer_conv):
                scale = int(layer_conv[1])
                d["maxpool2d{}".format(layer_idx)] = nn.MaxPool2d(
                    kernel_size=scale, stride=scale,
                    padding=0, dilation=1, ceil_mode=False)
            elif 'U' in str(layer_conv):
                scale = int(layer_conv[1])
                d["upsample2d{}".format(layer_idx)] = nn.Upsample(
                    scale_factor=scale, mode=upsample_mode, align_corners=True)
            elif 'S' in str(layer_conv):
                scale = int(layer_conv[1])
                d["shuffle2d{}".format(layer_idx)] = nn.PixelShuffle(scale)
                prev_conv = int(prev_conv/(scale*scale))
            else:
                #print(prev_conv, layer_conv)
                if depth:
                    #hidden = int(prev_conv*3)
                    hidden = 260
                    d["conv1{}".format(layer_idx)] = ConvBNReLU(prev_conv, layer_conv, kernel_size=1, groups=layer_conv)
                    #d["conv2{}".format(layer_idx)] = ConvBNReLU(hidden, hidden, kernel_size=1, groups=hidden)
                    #d["conv3{}".format(layer_idx)] = nn.Conv2d(hidden, layer_conv, kernel_size=1,
                    #                                                    stride=1, padding=0, bias=False)
                    d["relu{}".format(layer_idx)] = nn.ReLU(inplace=True)
                    prev_conv = networklist[layer_idx]
                else:
                    d["conv{}".format(layer_idx)] = misc_nn_ops.Conv2d(
                        prev_conv, layer_conv, kernel_size=kernel_size,
                        stride=1, padding=padding_n)
                    d["relu{}".format(layer_idx)] = nn.ReLU(inplace=True)
                    prev_conv = networklist[layer_idx]
            #d["upsample_conv_2d{}".format(layer_idx)] = nn.ConvTranspose2d(prev_conv,
            #    layer_conv, 3, stride=1, padding=(1,1))
        if output_activation == 'sigmoid':
            d.popitem()
            d['sigmoid'] = nn.Sigmoid()
        elif output_activation == 'batchnorm':
            d.popitem()
            d['bn'] = nn.BatchNorm2d(prev_conv)
        elif output_activation == 'remove':
            d.popitem()
        elif output_activation == 'logsoftmax':
            d.popitem()
            d['logsoftmax'] = nn.LogSoftmax(dim=1)
        d = [l for k,l in d.items()]
        if encoder_hook_index is not None:
            register_layers(d, encoder_hook_index, 'student_encoder')
        if decoder_hook_index is not None:
            register_layers(d, decoder_hook_index, 'student_decoder')

        super(CNN_constructer, self).__init__(*d)
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.normal_(param, std=0.01)
                # nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            elif "bias" in name:
                nn.init.constant_(param, 0)

class adaptation_layers(nn.Module):
    def __init__(self, layers_config, kernel_n =3, padding_n=1, normalize=None, detach=False, depth=False):
        super(adaptation_layers, self).__init__()
        self.layers = nn.ModuleList()
        self.detach = detach
        for i, ls in enumerate(layers_config):
            if isinstance(kernel_n, int):
                kn = kernel_n; pn = padding_n
            else:
                kn = kernel_n[i]; pn = padding_n[i]
            if isinstance(ls, int):
                block = nn.BatchNorm2d(ls)
            else:
                #print('here', normalize)
                block = CNN_constructer(ls, kernel_size=kn, padding_n=pn, depth=depth)
                if normalize is not None:
                    block = CNN_constructer(ls, output_activation=normalize, kernel_size=kn, padding_n=pn,  depth=depth)
            self.layers.append(
                block
            )
        self.__init_weight()
        #exit()

    def forward(self, inputs, resize=None):
        out = []
        n_layer = 0
        for input_x, layer in zip(inputs, self.layers):
            if self.detach: input_x = input_x.detach()
            #print(input_x.shape)
            tmp = layer(input_x)
            #print(tmp.shape)
            if resize is not None and resize[n_layer] != 1:
                tmp = nn.functional.interpolate(tmp, scale_factor=resize[n_layer],
                                                mode='bilinear', align_corners=True)
            out.append(tmp)
            n_layer += 1
        if resize is not None:
            out = t.cat(out, dim=1)
            out,_ = t.max(out, dim=1, keepdim=True)
        return out


    def __init_weight(self, std=0.01):
        import math
        for name, param in self.named_parameters():
            #print(name)
            if "weight" in name and 'conv' in name:
                #print('conv_weights', name)
                n = param.shape[2] * param.shape[3] * param.shape[1]
                nn.init.normal_(param, mean=0, std=math.sqrt(2. / n))
                #nn.init.normal_(param, std=std)
                # nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            elif 'weight' in name and 'conv' not in name:
                nn.init.normal_(param, std=std)
            elif "bias" in name:
                nn.init.constant_(param, 0)