import torch.nn as nn
from torchvision.models.mobilenet import InvertedResidual, ConvBNReLU

from .adaptation_layer import adaptation_layers
from .utils import mobilenetv2_pretrain

def generate_model9(pretrain_mode, student_model=None, state_dict=None, forward_hook_index=None):
    if state_dict is not None:
        student_model.init_pretrain()
        assert state_dict is not None, 'need state dict to switch to fine_tune mode'
        student_model.load_state_dict(state_dict)
        student_model.pretrain_mode = False
        adaptation_config = [(28, 128), (24, 256, 'U2'), (136, 512, 'U4'), (520, 512, 'U8')]
        adapter = adaptation_layers(adaptation_config, kernel_n=1, padding_n=0, depth=False)
        #merger = adaptation_layers([(1408, 'S2', 1)], kernel_n=3, padding_n=1, normalize='sigmoid')
        merger = adaptation_layers([(1408, 'S2', 1)], kernel_n=3, padding_n=1, normalize='remove')  # model 9 deepgazenpy
        return student_model.ecer, adapter, merger
    else:
        encoder = mobilenetv2_pretrain(forward_hook_index)
        if pretrain_mode:
            adaptation_config = [(28, 128), (24, 256), (136, 512), (520, 512)] #model 9
            adapter = adaptation_layers(adaptation_config, kernel_n=1, padding_n=0, depth=False) #model 9
            return encoder, adapter
        else:
            adaptation_config = [(28, 128), (24, 256, 'U2'), (136, 512, 'U4'), (520, 512, 'U8')]
            adapter = adaptation_layers(adaptation_config, kernel_n=1, padding_n=0, depth=False) #model 9
            #merger = adaptation_layers([(1408, 'S2', 1)], kernel_n=3, padding_n=1, normalize='sigmoid') #model 9
            merger = adaptation_layers([(1408, 'S2', 1)], kernel_n=3, padding_n=1, normalize='remove')  # model 9 deepgazenpy
            return encoder, adapter, merger

def init(model):
    for name, param in model.named_parameters():
        if "weight" in name:
            nn.init.normal_(param, std=0.01)
            # nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
        elif "bias" in name:
            nn.init.constant_(param, 0)
class Shared_inverted_block(nn.Module):
    def __init__(self, input_filter, output_filter, stride, expansion, convbnrele=False):
        super(Shared_inverted_block, self).__init__()
        self.inv_block = InvertedResidual(input_filter, output_filter, stride, expansion)
        if convbnrele:
            self.inv_block = ConvBNReLU(input_filter, output_filter, stride=stride)
        #self.drop = nn.Dropout2d(0.5)
        init(self)

    def forward(self, x, prev_x=None):
        if prev_x is not None:
            x = x + prev_x
        x1 = self.inv_block(x)
        x = x + x1
        return x

def generate_model10(pretrain_mode, student_model=None, state_dict=None, forward_hook_index=None):
    if state_dict is not None:
        student_model.init_pretrain()
        assert state_dict is not None, 'need state dict to switch to fine_tune mode'
        student_model.load_state_dict(state_dict)
        student_model.pretrain_mode = False
        adaptation_config = [(28, 8), (24, 32), (136, 128), (520, 512)]
        adapter = adaptation_layers(adaptation_config, kernel_n=1, padding_n=0, depth=False)

        s_block1 = Shared_inverted_block(8, 8, 1, 2)
        s_block2 = Shared_inverted_block(32, 32, 1, 2)
        s_block3 = Shared_inverted_block(128, 128, 1, 2)
        s_block4 = Shared_inverted_block(512, 512, 1, 2)

        #merger = adaptation_layers([(2, 1)], kernel_n=3, padding_n=1, normalize='sigmoid')
        merger = adaptation_layers([(2, 1)], kernel_n=3, padding_n=1, normalize='remove')
        return student_model.ecer, adapter, s_block1, s_block2, s_block3, s_block4, merger
    else:
        encoder = mobilenetv2_pretrain(forward_hook_index)
        if pretrain_mode:
            adaptation_config = [(28, 128), (24, 256), (136, 512), (520, 512)] #model 9
            adapter = adaptation_layers(adaptation_config, kernel_n=1, padding_n=0, depth=False) #model 9
            return encoder, adapter
        else:
            adaptation_config = [(28, 8), (24, 32), (136, 128), (520, 512)]
            adapter = adaptation_layers(adaptation_config, kernel_n=1, padding_n=0, depth=False)

            s_block1 = Shared_inverted_block(8, 8, 1, 2)
            s_block2 = Shared_inverted_block(32, 32, 1, 2)
            s_block3 = Shared_inverted_block(128, 128, 1, 2)
            s_block4 = Shared_inverted_block(512, 512, 1, 2)

            #merger = adaptation_layers([(2, 1)], kernel_n=3, padding_n=1, normalize='sigmoid')
            merger = adaptation_layers([(2, 1)], kernel_n=3, padding_n=1, normalize='remove')
            return encoder, adapter, s_block1, s_block2, s_block3, s_block4, merger