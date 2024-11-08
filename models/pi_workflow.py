import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PIModel(nn.Module):
    def __init__(self, base_model, in_channels, classifier):
        super(PIModel, self).__init__()
        # self.n_classes = num_classes
        # self.fc_input_dim = fc_input_dim
        # modules = list(base_model.children())
        new_conv_layer = nn.Conv1d(in_channels, in_channels, 3, padding=1)
        base_model.append(new_conv_layer)
        # modules[-1] = nn.Sequential(nn.Flatten(),
        #                             nn.Linear(in_features=self.fc_input_dim, out_features=self.n_classes))
        self.backbone = nn.Sequential(*base_model)
        self.classifier = classifier

        # Correlation matrix for multi-label (still used but for independent labels)
        # self.correlation = nn.Parameter(F.softmax(torch.rand(size=(self.n_classes, self.n_filters)), dim=0))

    def multinomial(self, pred_softmax, target):
        sample_cat_oh = torch.zeros(pred_softmax.shape)
        for i in range(len(sample_cat_oh)):
            sample_cat_oh[i] = F.one_hot(torch.multinomial(pred_softmax[i], target[i].sum(), replacement=False).squeeze(), 
                                         num_classes=target.shape[1])
        return sample_cat_oh
    
    def calc_pi_acc(self, pred, target, features):
        device = next(self.parameters()).device
        pred_1 = pred  # prediction of discrimination pathway (using all filters)

        # sample class ID using reparameter trick
        # pred_softmax = torch.softmax(pred, dim=-1)
        # with torch.no_grad():
        #     sample_cat = torch.multinomial(pred, target.sum(1), replacement=False).flatten().cuda()
        #     # sample_cat_oh = self.multinomial(pred, target).to(device)
        #     ind_positive_sample = sample_cat_oh == target
        #     sample_cat_oh = sample_cat.float().to(device)   # One-hot is not needed since it's multi-label
        # #   ind_positive_sample = sample_cat == target  # mark wrong sample results
        #     # sample_cat_oh = F.one_hot(sample_cat, num_classes=pred.shape[1]).float().cuda()
        #     epsilon = torch.where(sample_cat_oh != 0, 1 - pred, -pred).detach()
        # sample_cat_oh = pred + epsilon

        # # sample filter using reparameter trick
        # correlation_softmax = F.softmax(self.correlation, dim=0)
        # correlation_samples = sample_cat_oh @ correlation_softmax
        # with torch.no_grad():
        #     ind_sample = torch.bernoulli(correlation_samples).bool()
        #     epsilon = torch.where(ind_sample, 1 - correlation_samples, -correlation_samples)
        # binary_mask = correlation_samples + epsilon
        # feature_mask = features * binary_mask[..., None]  # binary
        # pred_2 = self.classifier(feature_mask) # prediction of Interpretation pathway (using a cluster of class-specific filters)
        # with torch.no_grad(): 
        #     correlation_samples = correlation_softmax[target]
        #     binary_mask = torch.bernoulli(correlation_samples).sum(1).bool()
        #     feature_mask_self = features * ~binary_mask[..., None]
        #     pred_3 = self.classifier(feature_mask_self) # prediction of Interpretation pathway (using complementary clusters of class-specific filters)
        # out = {"features": features, 'pred_1': pred_1, 'pred_2': pred_2, 'pred_3': pred_3,
        #             'ind_positive_sample': ind_positive_sample}
        # return out

        correlation = nn.Parameter(F.softmax(torch.rand(size=(target.shape[-1], features.shape[1])), dim=0)).to(device)
        # sample class ID using reparameter trick (adjusted for multi-label)
        with torch.no_grad():
            sample_cat = torch.bernoulli(pred).to(device)  # Sampling binary labels for multi-label case
            ind_positive_sample = sample_cat == target  # mark correct samples
            sample_cat_oh = sample_cat.float().to(device)   # One-hot is not needed since it's multi-label
            epsilon = torch.where(sample_cat_oh != 0, 1 - pred, -pred).detach()

        # Adjust for binary mask
        sample_cat_oh = pred + epsilon

        # sample filter using reparameter trick (adjusted for multi-label)
        correlation_softmax = F.softmax(correlation, dim=0) # correlation
        correlation_samples = sample_cat_oh @ correlation_softmax
        with torch.no_grad():
            ind_sample = torch.bernoulli(correlation_samples).bool()
            epsilon = torch.where(ind_sample, 1 - correlation_samples, -correlation_samples)

        binary_mask = correlation_samples + epsilon
        feature_mask = features * binary_mask[..., None]  # binary mask applied to features

        # Prediction using class-specific filters
        pred_2 = torch.sigmoid(self.classifier(feature_mask))  # Sigmoid for multi-label prediction

        # Complementary filters sampling
        with torch.no_grad():
            correlation_samples = correlation_softmax[target.int()]
            binary_mask = torch.bernoulli(correlation_samples).bool()
            binary_mask_squeezed = torch.any(binary_mask, dim=1)
            # features = features.unsqueeze(1).repeat(1,self.n_classes,1,1,1)
            feature_mask_self = features * ~binary_mask_squeezed[..., None]

        pred_3 = torch.sigmoid(self.classifier(feature_mask_self))  # Sigmoid for multi-label prediction

        # Output dictionary with the predictions and features
        return {"features": features, 'pred_1': pred_1, 'pred_2': pred_2, 'pred_3': pred_3,
               'ind_positive_sample': ind_positive_sample
               }
            
    def forward(self, inputs, target=None, forward_pass='default'):
        features = self.backbone(inputs)
        pred = torch.sigmoid(self.classifier(features))  # Use sigmoid for multi-label classification
        accdict = self.calc_pi_acc(pred, target, features)
        return accdict
    #     pred_1 = pred  # prediction of discrimination pathway (using all filters)

    #     # sample class ID using reparameter trick (adjusted for multi-label)
    #     with torch.no_grad():
    #         sample_cat = torch.bernoulli(pred).to(device)  # Sampling binary labels for multi-label case
    #         ind_positive_sample = sample_cat == target  # mark correct samples
    #         sample_cat_oh = sample_cat.float().to(device)   # One-hot is not needed since it's multi-label
    #         epsilon = torch.where(sample_cat_oh != 0, 1 - pred, -pred).detach()

    #     # Adjust for binary mask
    #     sample_cat_oh = pred + epsilon

    #     # sample filter using reparameter trick (adjusted for multi-label)
    #     correlation_softmax = F.softmax(self.correlation, dim=0)
    #     correlation_samples = sample_cat_oh @ correlation_softmax
    #     with torch.no_grad():
    #         ind_sample = torch.bernoulli(correlation_samples).bool()
    #         epsilon = torch.where(ind_sample, 1 - correlation_samples, -correlation_samples)

    #     binary_mask = correlation_samples + epsilon
    #     feature_mask = features * binary_mask[..., None, None]  # binary mask applied to features

    #     # Prediction using class-specific filters
    #     pred_2 = torch.sigmoid(self.classifier(feature_mask))  # Sigmoid for multi-label prediction

    #     # Complementary filters sampling
    #     with torch.no_grad():
    #         correlation_samples = correlation_softmax[target]
    #         binary_mask = torch.bernoulli(correlation_samples).bool()
    #         binary_mask_squeezed = torch.any(binary_mask, dim=1)
    #         # features = features.unsqueeze(1).repeat(1,self.n_classes,1,1,1)
    #         feature_mask_self = features * ~binary_mask_squeezed[..., None, None]

    #     pred_3 = torch.sigmoid(self.classifier(feature_mask_self))  # Sigmoid for multi-label prediction

    #     # Output dictionary with the predictions and features
    #     out = {"features": features, 'pred_1': pred_1, 'pred_2': pred_2, 'pred_3': pred_3,
    #            'ind_positive_sample': ind_positive_sample}
    #     return out

def create_config(config_file, backbone=None, criterion=None):
    import os
    from datetime import datetime
    import yaml
    from easydict import EasyDict
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    cfg = EasyDict()

    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    cfg = EasyDict()

    # Copy
    for k, v in config.items():
        cfg[k] = v
    if backbone:
        cfg['backbone'] = backbone
    if criterion:
        cfg['criterion'] = criterion

    # Set paths for pretext task (These directories are needed in every stage)
    if not os.path.exists('./result'):
        os.mkdir('./result')
    dt = datetime.now().strftime('%m%d_%H%M')
    base_dir = f'./result/{dt}_' + cfg['db_name'] + f"_{cfg['backbone']}"+ f"_{cfg['criterion']}"
    if os.path.exists(base_dir):
        for i in range(10):
            base_dir = base_dir + f'_{i + 2}'
            if os.path.exists(base_dir):
                continue
            break
    os.mkdir(base_dir)

    cfg['base_dir'] = base_dir
    cfg['best_checkpoint'] = os.path.join(base_dir, 'best_checkpoint.pth.tar')
    cfg['last_checkpoint'] = os.path.join(base_dir, 'last_checkpoint.pth.tar')
    return cfg


if __name__ == '__main__':
    import argparse
    from MLPMixer import resnet
    cmd_opt = argparse.ArgumentParser(description='Argparser for PICNN')
    cmd_opt.add_argument('-configFileName', default='./configs/cifar10.yml')
    cmd_opt.add_argument('-criterion', default='ClassSpecificCE',help='StandardCE/ClassSpecificCE')
    cmd_opt.add_argument('-backbone', default='resnet18',help='resnet18')
    cmd_args, _ = cmd_opt.parse_known_args()
    p = create_config(config_file=cmd_args.configFileName,
                      backbone=cmd_args.backbone,
                      criterion=cmd_args.criterion)   
    input = torch.randn((128, 3, 224, 224)).cuda()
    # target = F.one_hot(torch.randint(0, 10, (128,)), num_classes=10).cuda()
    a=np.array([[0,1,0,1,0,0,0,0,0,0,], [1,0,0,0,0,0,0,0,1,0], [0,1,1,1,1,1,0,0,0,0,],[0,0,0,0,0,0,1,0,1,1,]])
    target = []
    for i in range(32):
        target.append(a)
    target = torch.tensor(np.vstack(target)).cuda()
    model = PIModel(p).cuda()
    model(input, target)