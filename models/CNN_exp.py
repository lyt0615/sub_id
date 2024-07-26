import torch
import torch.nn as nn
import torch.nn.functional as F

def get_fc_param(input_size, conv_param, fc_layers, dropout=None):
    for params in conv_param:
        input_size[1] = (input_size[1]+2*params['conv_padding']-params['conv_ksize'])//params['conv_stride']+1
        input_size[0] = params['conv_cout']
        if params['mp_ksize'] is not None and params['mp_stride'] is not None:
            input_size[1] = (input_size[1]-params['mp_ksize'])//params['mp_stride']+1
    return {'input_dim': input_size[0]*input_size[1], 'num_layers': fc_layers, 'dropout': dropout}

class CNN_exp(nn.Module):
    def __init__(self, conv_param, fc_param, class_num=37):
        super(CNN_exp, self).__init__()

        # convolutional layers
        conv_layers = []
        for param in conv_param:
            if param['mp_ksize'] is not None and param['mp_stride'] is not None:  
                block = [nn.Conv1d(param['conv_cin'], param['conv_cout'], kernel_size=param['conv_ksize'], stride=param['conv_stride'], padding=param['conv_padding']),
                        nn.MaxPool1d(kernel_size=param['mp_ksize'], stride=param['mp_stride']),
                        nn.BatchNorm1d(param['conv_cout']),
                        nn.ReLU()
                        ]
            else: 
                block = [nn.Conv1d(param['conv_cin'], param['conv_cout'], kernel_size=param['conv_ksize'], stride=param['conv_stride'], padding=param['padding']),
                        nn.BatchNorm1d(param['conv_cout']),
                        nn.ReLU()
                        ]
            conv_layers.extend(block)
        self.conv_layers = nn.Sequential(*conv_layers)

        # fc layers
        self.fc= create_mlp_block(fc_param['input_dim'], output_dim=class_num, num_layers=fc_param['num_layers'])

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
def create_mlp_block(input_dim, output_dim, num_layers):
    layers = []
    current_dim = input_dim
    interval = (input_dim - output_dim) // num_layers
    
    for i in range(num_layers):
        if i != num_layers-1:
            next_output_dim = input_dim - (i+1) * interval
            layers.append(nn.Linear(current_dim, next_output_dim))
            layers.append(nn.ReLU())
            current_dim = next_output_dim
        else: 
            next_output_dim = output_dim
            layers.append(nn.Linear(current_dim, next_output_dim))
    return nn.Sequential(*layers)

if __name__ == '__main__':

    conv_param = [{'conv_cin': 1, 'conv_cout': 32, 'conv_ksize': 5, 'conv_stride': 2, 'conv_padding': 1, 'mp_ksize': 2, 'mp_stride': 2,},
                    {'conv_cin': 32, 'conv_cout': 64, 'conv_ksize': 5, 'conv_stride': 2, 'conv_padding': 1, 'mp_ksize': 2, 'mp_stride': 2,},
                    {'conv_cin': 64, 'conv_cout': 128, 'conv_ksize': 5, 'conv_stride': 1, 'conv_padding': 1, 'mp_ksize': 2, 'mp_stride': 2,},
                    {'conv_cin': 128, 'conv_cout': 256, 'conv_ksize': 5, 'conv_stride': 1, 'conv_padding': 1, 'mp_ksize': 2, 'mp_stride': 2,},
                    # {'conv_cin': 256, 'conv_cout': 512, 'conv_ksize': 5, 'conv_stride': 1, 'conv_padding': 1, 'mp_ksize': 2, 'mp_stride': 2,},
                    ]
    # fc_param = [{'input_dim': 1, 'num_layers': 4,}]

    input_size = [1, 1024]
    fc_param = get_fc_param(input_size, conv_param, 4)
    model = CNN_exp(conv_param, fc_param)
    output = model(torch.randn(1, 1024))
    print(output.shape)
    print(model)
