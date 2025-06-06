import logging
from tqdm import tqdm
import numpy as np
import random
import os
import torch
from sklearn import metrics
from utils.dataloader import make_trainloader, make_testloader
import torch.nn as nn
import torch.nn.functional as F

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class ClassSpecificCE(nn.Module):
    def __init__(self, criterion, lambda_weight=0):
        super(ClassSpecificCE, self).__init__()
        self.lambda_weight = lambda_weight
        self.criterion = criterion

    def forward(self, inputs, targets):
        pred_1 = inputs['pred_1']
        pred_2 = inputs['pred_2']
        pred_3 = inputs['pred_3']
        ind_positive_sample = inputs['ind_positive_sample']
        n_positive_sample = int(torch.sum(ind_positive_sample))
        # with torch.no_grad():
        #     ACC1 = int((pred_1.argmax(-1) == targets).sum()) / targets.shape[0]
        #     ACC2 = int((pred_2.argmax(-1) == targets).sum()) / targets.shape[0]
        #     ACC3 = int((pred_3.argmax(-1) == targets).sum()) / targets.shape[0]
        loss_discrimination = self.criterion(pred_1, targets)  # included 'softmax'
        if n_positive_sample != 0:
            loss_interpretation = self.criterion(pred_2[ind_positive_sample], targets[ind_positive_sample])
        else:
            loss_interpretation = torch.tensor(0.0).cuda()
        loss_total = loss_discrimination + loss_interpretation * self.lambda_weight
        return {"loss_discrimination": loss_discrimination, "loss_interpretation": loss_interpretation, "loss_total": loss_total,
                "n_positive_sample": n_positive_sample, 
                # 'ACC1': ACC1, 'ACC2': ACC2,'ACC3': ACC3
                }
    
class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStop:
    def __init__(self, patience=10, mode='max', delta=0.0001):
        self.patientce = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == 'min':
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):
        if self.mode == 'min':
            score = -1. * epoch_score
        else:
            score = np.copy(epoch_score)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patientce:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score


class Engine:
    def __init__(self, train_loader=None, val_loader=None, test_loader=None,
                 criterion=None, optimizer=None, scheduler=None,
                 model=None, device='cpu', use_pi=0):

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_pi = use_pi

        self.model = model
        self.device = device

    def forward_backward(self, data, target, losses, binary, train=True):
        output = self.model(data, target) if self.use_pi else self.model(data)
        if binary:
            if not self.use_pi: output = torch.sigmoid(output)
        loss = self.criterion(output, target)
        loss = loss['loss_total'] if self.use_pi else loss
        if train:
            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
        losses.update(loss.item(), target.size(0))
        return losses, output
        
    def train_epoch(self, epoch, binary=False):

        losses = AverageMeter()
        self.model.train()

        bar = tqdm(self.train_loader)
        for _, (data, target) in enumerate(bar):

            data, target = data.to(self.device), target.to(self.device)
            losses, output = self.forward_backward(data, target, losses, binary)

            # self.optimizer.zero_grad()
            # output = self.model(data, target) if self.use_pi else self.model(data)
            # if binary:
                # output = torch.sigmoid(output)

            # loss = self.criterion(output, target) 
            # loss.backward() 
            # self.optimizer.step()

            # losses.update(loss.item(), target.size(0))
            bar.set_description(
                f'Epoch{epoch:3d}, train loss:{losses.avg:6f}')

        logging.info(f'Epoch{epoch:3d}, train loss:{losses.avg:6f}')
        return losses.avg, output

    def evaluate_epoch(self, epoch, binary=False):

        accuracies = AverageMeter()
        losses = AverageMeter()
        flag = True
        self.model.eval()
        bar = tqdm(self.val_loader)
        with torch.no_grad():
            for _, (data, target) in enumerate(bar):

                data, target = data.to(self.device), target.to(self.device)

                # output = self.model(data, target) if self.use_pi else self.model(data)
                # if binary:
                #     output = torch.sigmoid(output)
                # loss = self.criterion(output, target)
                # losses.update(loss.item(), target.size(0))
                losses, output = self.forward_backward(data, target, losses, binary, 0)
                output = output['pred_1']
                target = target.detach().cpu().numpy()

                if target.ndim == 1:
                    flag = True
                    prediction = torch.argmax(output, dim=1).view(-1).detach().cpu().numpy()
                    accuracy = (prediction == target).mean()
                    accuracies.update(accuracy.item(), data.size(0))
                    bar.set_description(
                        f'Epoch{epoch:3d}, valid loss:{losses.avg:6f} , accuracy:{accuracies.avg:6f}')
                elif binary and len(set(target[:, 0])) == 2:
                    flag = True
                    prediction = torch.greater_equal(output, 0.5).to(torch.float64).cpu().detach().numpy()
                    accuracy = metrics.f1_score(target, prediction, average='macro', zero_division=0.0)
                    accuracies.update(accuracy.item(), data.size(0))
                    bar.set_description(
                        f'Epoch{epoch:3d}, valid loss:{losses.avg:6f} , valid accuracy:{accuracies.avg:6f}')
                else:
                    bar.set_description(
                        f'Epoch{epoch:3d}, valid loss:{losses.avg:6f}')

        if self.scheduler:
            self.scheduler.step()
        if target.ndim == 1:
            logging.info(f'Epoch{epoch:3d}, valid loss:{losses.avg:6f}, valid accuracy:{accuracies.avg:6f}')
        else:
            logging.info(f'Epoch{epoch:3d}, valid loss:{losses.avg:6f}')

        return losses.avg, accuracies.avg if flag else None

    def test_epoch(self, binary=False):

        accuracies = AverageMeter()
        losses = AverageMeter()

        self.model.eval()
        bar = tqdm(self.test_loader)
        outputs = []
        predicted = []
        true = []
        with torch.no_grad():
            for _, (data, target) in enumerate(bar):
                data, target = data.to(self.device), target.to(self.device)

                losses, output = self.forward_backward(data, target, losses, binary, 0)
                loss = losses.val
                output = output['pred_2']

                target = target.detach().cpu().numpy()
                if binary:
                    prediction = torch.greater_equal(output, 0.5).to(torch.float64).cpu().detach().numpy()
                    accuracy = metrics.f1_score(target, prediction, 
                                                average='macro', zero_division=0.0)
                else:
                    prediction = torch.argmax(output, dim=1).view(-1).detach().cpu().numpy()
                    accuracy = (prediction == target).mean()

                outputs.append(output.detach().cpu().numpy())

                losses.update(loss.item(), data.size(0))
                accuracies.update(accuracy.item(), data.size(0))
                bar.set_description(
                    f"test loss: {losses.avg:.5f} accuracy:{accuracies.avg:.5f}")
                predicted += prediction.tolist()
                true += target.tolist()
            outputs = np.concatenate(outputs)
        return outputs, np.array(predicted), np.array(true), accuracies.avg, losses.avg


def load_net_state(net, state_dict):
    # check the keys and load the weight
    net_keys = net.state_dict().keys()
    state_dict_keys = state_dict.keys()
    for key in net_keys:
        if key in state_dict_keys:
            # load the weight
            net.state_dict()[key].copy_(state_dict[key])
        else:
            print('key error: ', key)
    net.load_state_dict(net.state_dict())
    return net


def train_model(model, save_path, ds, device='cpu', tune=False,  **kwargs):
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir = save_path)
    pool_dim = 256 if model.__class__.__name__ == 'Pace' else 1024

    train_loader, val_loader = make_trainloader(
        ds, batch_size=kwargs['batch_size'],
        num_workers=1, train_size=kwargs['train_size'],
        seed=42, tune=tune, pool_dim=pool_dim)

    test_loader = make_testloader(ds, pool_dim=pool_dim)
    binary = True if 'ir' in ds or 'raman' in ds else False
    if binary:
            criterion = torch.nn.BCELoss()
            if kwargs['use_pi']:
                criterion = ClassSpecificCE(criterion)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), **kwargs['Adam_params'])

    if ds == 'fcgformer_ir':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                                                  T_0=40,
                                                                                  T_mult=2)  # fcgformer
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    # mode = 'min' if not tune else 'max'
    mode = 'max'
    es = EarlyStop(patience=kwargs['patience'], mode=mode)

    engine = Engine(train_loader=train_loader, val_loader=val_loader,
                    test_loader=test_loader,
                    criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                    model=model, device=device, use_pi=kwargs['use_pi'])

    # start to train 
    for epoch in range(kwargs['epoch']):
        train_loss, output = engine.train_epoch(epoch, binary=binary)
        val_loss, acc = engine.evaluate_epoch(epoch, binary=binary)
        # _, _, _, test_acc, test_loss = engine.test_epoch(binary=binary)

        if acc:
            es(acc, model, f'{save_path}/{epoch}_f1_{str(acc)[2:6]}.pth')
        else:
            assert ValueError('not metrics')
        if es.early_stop:
            break
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", acc, epoch)
        writer.add_histogram('Pred/pred_1', output['pred_1'], epoch)
        writer.add_histogram('Pred/pred_2', output['pred_2'], epoch)
        # writer.add_scalar("Loss/test", test_loss, epoch)
        # writer.add_scalar("Accuracy/test", test_acc, epoch)
    print(f'Early stopped at: {es.val_score}.')
    torch.save(model.state_dict(), f'{save_path}/epoch{epoch}.pth')


def test_model(model, ds, device='cpu', verbose=True, **kwargs):
    pool_dim = 256 if model.__class__.__name__ == 'Pace' else 1024
    test_loader = make_testloader(ds, pool_dim=pool_dim)
    binary = True if 'ir' in ds or 'raman' in ds else False
    criterion = torch.nn.BCELoss() if binary else torch.nn.CrossEntropyLoss()
    engine = Engine(test_loader=test_loader,
                    criterion=criterion, model=model, device=device, use_pi=kwargs['use_pi'])
    outputs, pred, true, _, _ = engine.test_epoch(binary=binary)

    if verbose:
        from sklearn import metrics
        print(metrics.classification_report(true, pred, digits=4))
        print('Exact match rate (EMR): %.4f\n' %metrics.accuracy_score(true, pred))
        logging.info(metrics.classification_report(true, pred, digits=4))
        # logging.info(f'accuracy:{metrics.accuracy_score(true, pred):.5f}')
        logging.info('Exact match rate (EMR): %.4f\n' %metrics.accuracy_score(true, pred))
        logging.info('Accuracy: %.4f\n' %(np.count_nonzero(true==pred)/pred.shape[0]/pred.shape[1]))
    return outputs, pred, true

def inf_time(model):
    iterations = 300
    device = torch.device("cuda:0")
    model.to(device)

    random_input = torch.randn(1, 1, 1024).to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # Preheat GPU
    for _ in range(50):
        _ = model(random_input)

    # Measure inference time
    times = torch.zeros(iterations)     # Save the time of each iteration
    with torch.no_grad():
        for iter in range(iterations):
            starter.record()
            _ = model(random_input)
            ender.record()
            # Synchronize GPU time
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) # Calculate time
            times[iter] = curr_time
            # print(curr_time)

    mean_time = times.mean().item()
    return mean_time
    # print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000/mean_time))