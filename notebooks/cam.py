from Res_SE import resnet
import os
import torch.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIModules import ClassSpecificLoss



def compute_gradcam(model, input_data, target_class):
    # Forward pass

    if type(target_class) == int: target_class = [target_class]
    camlist = []
    for c in target_class:
        model.eval()
        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])
        activations, gradients = list(), list()
            
        # Register hooks
        # Register a hook on the target layer to save activations and gradients
        target_layer = list(model.modules())[1][0]    
        target_layer.eval()    
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_backward_hook(backward_hook)        
        output = model.pi_model(input_data, torch.randn(1,17))['pred_1']        
        score = output[:, c]  
        score.backward(retain_graph=True)
        forward_handle.remove()
        backward_handle.remove()        
        activation = activations[0].detach()  # Shape (batch, channels, length)
        gradient = gradients[0].detach()      # Shape (batch, channels, length)

        # Global Average Pooling on the gradients (temporal dimension)
        weights = torch.mean(gradient, dim=2, keepdim=True)  # Shape (batch, channels, 1)

        # Calculate weighted sum of activations
        gradcam_map = torch.sum(weights * activation, dim=1)  # Shape (batch, length)

        # Apply ReLU to retain only positive influence
        gradcam_map = nn.ReLU()(gradcam_map)

        # Normalize Grad-CAM map
        gradcam_map = gradcam_map - gradcam_map.min(dim=1, keepdim=True)[0]
        gradcam_map = gradcam_map / gradcam_map.max(dim=1, keepdim=True)[0]

        # Cleanup hooks

        camlist.append(gradcam_map)
    return gradcam_map  # Shape (batch, length)


x = np.load('/data/YantiLiu/projects/subs_id/datasets/fcgformer_ir/test_x.npy')
y = np.load('/data/YantiLiu/projects/subs_id/datasets/fcgformer_ir/test_y.npy')
id = 114
x, y = torch.tensor(x[id], dtype=torch.float32), y[id]

device='cuda:0'
params = {'conv_ksize':3, 
        'conv_padding':1, 
        'conv_init_dim':32, 
        'conv_final_dim':256, 
        'conv_num_layers':4, 
        'mp_ksize':2, 
        'mp_stride':2, 
        'fc_dim':1024, 
        'fc_num_layers':0, 
        'mixer_num_layers':6,
        'n_classes':17,
        'use_mixer':1,
        'use_se': 1,
        'use_res':1,
        'depth': 6,
        'use_pi': 1,
        # 'use_cam':1
        }
model = resnet(**params)
path = '/data/YantiLiu/projects/subs_id/checkpoints/fcgformer_ir/Res_SE/2024-11-10_17_50mixer6_layer6/194_f1_8484.pth'
model.load_state_dict(torch.load(path, map_location=device), strict=True)

cam = compute_gradcam(model, x.unsqueeze(0), y)
print(cam)
# Here is the code ：
# module_list = list(model.modules())
# model = model.pi_model
# target_layers = [module_list[-1]]
# cam = GradCAM(model=model, target_layers=target_layers)
# input_tensor = torch.randn(1,1,1024)
# # Prepare image
# img_path = "image.png"
# assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
# img = Image.open(img_path).convert('RGB')
# img = np.array(img, dtype=np.uint8)
# img_tensor = data_transform(img)
# input_tensor = torch.unsqueeze(img_tensor, dim=0)

# # Grad CAM
# targets = [ClassifierOutputTarget(1)]     # cat

# grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
# grayscale_cam = grayscale_cam[0, :]
