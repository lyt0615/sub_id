"""
@File        :main.py
@Author      :Xinyu Lu
@EMail       :xinyulu@stu.xmu.edu.cn
"""

import os
import time
import json
import logging
import argparse
import config
from utils.utils import seed_everything, load_net_state, train_model, test_model


def get_args_parser():
    parser = argparse.ArgumentParser('---', add_help=False)

    # basic params
    parser.add_argument('--net', default='CNN_exp',
                        help="Choose network")
    parser.add_argument('--ds', default='qm9s_raman',
                        help="Choose dataset")
    parser.add_argument('--device', default='cuda:0',
                        help="Choose GPU device")

    parser.add_argument('-train', '--train', action='store_true',
                        help="start train")
    parser.add_argument('-tune', '--tune', action='store_true',
                        help="start tune")
    parser.add_argument('-test', '--test', action='store_true',
                        help="start test")
    parser.add_argument('--debug', '-debug', action='store_true',
                        default=1,
                        help="start debug")


    parser.add_argument('--base_checkpoint',
                        help="Choose base model for fine-tune")
    parser.add_argument('--test_checkpoint',
                        help="Choose checkpoint for test")
    parser.add_argument('--seed',
                        default=2024,
                        help="Random seed")

    # params of CNN_exp
    parser.add_argument('--n_conv', 
                        help="Number of convolution layers in CNN_exp")
    parser.add_argument('--n_fc', 
                        help="Number of fc layers in CNN_exp")
    parser.add_argument('--n_mixer', 
                        help="Number of MLPMixer1D")   
    parser.add_argument('--use_mixer', default=False,
                        help="Use MLPMixer1D or not")  
    
    # params of strategy
    parser.add_argument('--train_size',
                        help="train size for train_val_split")
    parser.add_argument('--batch_size',
                        help="batch size for training")
    parser.add_argument('--epoch',
                        help="epochs for training")
    parser.add_argument('--lr',
                        help="learning rate")
    args = parser.parse_args()
    return args

def catch_exception():
    import traceback
    import shutil

    traceback.print_exc()
    
    if os.path.exists(f'logs/{ds}/{net_}/{ts}_{mode}.log'):
        os.remove(f'logs/{ds}/{net_}/{ts}_{mode}.log') 
        print('unexpected log has been deleted')
    if os.path.exists(f'checkpoints/{ds}/{net_}/{ts}'):
        shutil.rmtree(f'checkpoints/{ds}/{net_}/{ts}')
        print('unexpected tensorboard record has been deleted')

if __name__ == "__main__":

    # for f in config.FC:
    #     for c in config.CONVS:
    #         n_conv, n_fc = c, f
    args = get_args_parser()

    if args.train and args.ds == 'Bacteria':
        args.lr = '1e-3'
        args.epoch = '50'

    if args.tune and args.ds == 'Bacteria':
        args.batch_size = '8'

    seed_everything(int(args.seed))
    
    params = {'net': config.NET, 'strategy': config.STRATEGY['train'] if args.train or args.debug else config.STRATEGY['tune']}
    params['net']['use_mixer'] = eval(args.use_mixer) if type(args.use_mixer) == str else args.use_mixer

    if args.n_conv:
        params['net']['conv_num_layers'] = int(args.n_conv)
    if args.n_fc:
        params['net']['fc_num_layers'] = int(args.n_fc)
    if args.n_mixer:
        params['net']['mixer_num_layers'] = int(args.n_mixer)
        

    if args.batch_size:
        params['strategy']['batch_size'] = int(args.batch_size)
    if args.epoch:
        params['strategy']['epoch'] = int(args.epoch)
    if args.lr:
        params['strategy']['Adam_params']["lr"] = float(args.lr)
    if args.train_size:
        params['strategy']['train_size'] = float(args.train_size)

    if args.net == 'CNN_exp':
        n_conv = params['net']['conv_num_layers']
        
        if args.use_mixer:
            n_mixer = params['net']['mixer_num_layers']
            ts = time.strftime('%Y-%m-%d_%H_%M', time.localtime()) + f'_conv{n_conv}mixer{n_mixer}'
        else:
            n_fc = params['net']['fc_num_layers']
            ts = time.strftime('%Y-%m-%d_%H_%M', time.localtime()) + f'_conv{n_conv}fc{n_fc}'
        
        # assert n_fc is None and args.use_mixer is None, 'Only one block should be used: either MLP or MLPMixer.'
            
    elif args.net == 'ResPeak_MLPMixer1D' or args.net == 'CNN_MLPMixer1D':
        if args.use_mixer:
            ts = time.strftime('%Y-%m-%d_%H_%M', time.localtime()) + f'_{int(args.n_mixer)}'
    
    else:
        ts = time.strftime('%Y-%m-%d_%H_%M', time.localtime())

    ds = args.ds
    net_ = args.net
    device = args.device

    if not os.path.exists('logs'):
        os.mkdir('logs')
    if not os.path.exists(f'logs/{ds}'):
        os.mkdir(f'logs/{ds}')
    if not os.path.exists(f'logs/{ds}/{net_}'):
        os.mkdir(f'logs/{ds}/{net_}')

    if args.train or args.debug:
        mode = "train"
    elif args.tune:
        mode = "tune"
    elif args.test:
        mode = "test"

    logging.basicConfig(
        filename=f'logs/{ds}/{net_}/{ts}_{mode}.log',
        format='%(levelname)s:%(message)s',
        level=logging.INFO,
    )

    logging.info({k: v for k, v in args.__dict__.items() if v})

    # ================================2. data & params ======================================
    data_root = f"datasets/{ds}"
    save_path = f"checkpoints/{ds}/{net_}/{ts}"

    if args.train:
        if not os.path.exists(f"checkpoints/{ds}"):
            os.mkdir(f"checkpoints/{ds}")
        if not os.path.exists(f"checkpoints/{ds}/{net_}"):
            os.mkdir(f"checkpoints/{ds}/{net_}")
        if not os.path.exists(f"checkpoints/{ds}/{net_}/{ts}"):
            os.mkdir(f"checkpoints/{ds}/{net_}/{ts}")

    if args.ds == 'nist_ir' or args.ds == 'fcgformer_ir':
        n_classes = 17
    elif 'qm9s' in args.ds:
        n_classes = 957
    else:
        n_classes = len(json.load(open(os.path.join(data_root, 'label.json'))))


    if net_ == 'VanillaTransformer':
        from models.VanillaTransformer import VanillaTransformerEncoder
        net = VanillaTransformerEncoder(**params['net']).to(device)    
    elif net_ == 'ResNet':
        from models.ResNet import resnet
        net = resnet(**params['net']).to(device)
    elif net_ == 'LSTM':
        from models.LSTM import LSTM
        net = LSTM(n_classes=n_classes).to(device)
    elif net_ == 'TextCNN':
        from models.TextCNN import TextCNN
        net = TextCNN(n_classes=n_classes).to(device)

    elif net_ == 'RamanNet':
        from models.RamanNet import RamanNet
        net = RamanNet(n_classes=n_classes).to(device)

    elif net_ == 'ConvMSANet':
        from models.ConvMSANet import convmsa_reflection
        net = convmsa_reflection(stem=True, **params['net']).to(device)

    elif net_ == 'PACE':
        from models.PACE import Pace
        net = Pace(n_classes=n_classes).to(device)

    elif net_ == 'ConvNext':
        from models.ConvNext import ConvNeXt
        net = ConvNeXt(**params['net']).to(device)

    elif net_ == 'Xception':
        from models.Xception import xception
        net = xception(n_classes).to(device)

    elif net_ == 'SANet':
        from models.SANet import SANet
        net = SANet(n_classes).to(device)

    elif net_ == 'CNN':
        from models.CNN import CNN
        net = CNN(n_classes).to(device)

    elif net_ == 'ResPeak':
        from models.ResPeak import resunit
        net = resunit(**params['net']).to(device)
        print(net)

    elif net_ == 'CNN_exp':
        from models.CNN_exp import CNN_exp
        net = CNN_exp(**params['net']).to(device)

    logging.info(net)

    # ================================3. start to train/tune/test ======================================
    try:

        if args.train or args.debug:
            train_model(net, save_path, ds=args.ds, device=device, **params['strategy'])

        elif args.tune:
            import torch

            base_model_path = args.base_model_path
            print(base_model_path)
            net = load_net_state(net, torch.load(f'{base_model_path}.pth'))
            train_model(net, save_path=f"{base_model_path}/tune", ds=args.ds, device=device,
                        tune=True, **params['strategy'])

        elif args.test:
            import torch

            test_model_path = args.test_checkpoint
            print(test_model_path)
            net = load_net_state(net, torch.load(test_model_path,
                                                map_location={'cuda:0': device, 'cuda:1': device}))
            test_model(net, device=device, ds=args.ds)

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(e)
        os.remove(f'logs/{ds}/{net_}/{ts}_{mode}.log')



