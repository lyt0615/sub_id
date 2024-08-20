NET = {'conv_ksize':3, 
       'conv_padding':1, 
       'conv_init_dim':32, 
       'conv_final_dim':256, 
       'conv_num_layers':4, 
       'mp_ksize':2, 
       'mp_stride':2, 
       'fc_dim':1024, 
       'fc_num_layers':4, 
       'mixer_num_layers':4,
       'n_classes':957,
    #    'use_mixer':True
       }


STRATEGY = {
    'train': {
        "batch_size": 64,
        "epoch": 200,
        "patience": 200,
        'train_size': None,
        "optmizer": "Adam",
        "Adam_params": {"lr": 1e-4}, 
        # "Adam_params": {"lr": 1e-3}, # for Bacteria
    },
    # 'tune': {
    #     "batch_size": 64,
    #     # "batch_size": 8, # for Bacteria
    #     "epoch": 200,
    #     "patience": 50,
    #     'train_size': None,
    #     "optmizer": "Adam",
    #     "Adam_params": {"lr": 1e-5},
    # }
}

CONVS = range(3, 8)
FC = range(3, 5)