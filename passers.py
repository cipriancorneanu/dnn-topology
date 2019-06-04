from utils import progress_bar
import torch
import numpy as np
from labels import identity


def get_accuracy(predictions, targets):
    ''' Compute accuracy of predictions to targets. max(predictions) is best'''
    _, predicted = predictions.max(1)
    total = targets.size(0)
    correct = predicted.eq(targets).sum().item()

    return 100.*correct/total


class Passer():
    def __init__(self, net, loader, criterion, device, repeat=1):
        self.network = net
        self.criterion = criterion
        self.device = device
        self.loader = loader
        self.repeat = repeat

    def _pass(self, optimizer=None, manipulator=identity, mask=None):
        ''' Main data passing routing '''
        losses, features, total, correct = [], [], 0, 0
        accuracies = []
        
        for r in range(1, self.repeat+1):
            for batch_idx, (inputs, targets) in enumerate(self.loader):
                targets = manipulator(targets)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
            
                if optimizer: optimizer.zero_grad()
                if mask:
                    outputs = self.network(inputs, mask)
                else:
                    outputs = self.network(inputs)

                loss = self.criterion(outputs, targets)
                losses.append(loss.item())

                if optimizer:
                    loss.backward()
                    optimizer.step()

                
                accuracies.append(get_accuracy(outputs, targets))
                progress_bar((r-1)*len(self.loader)+batch_idx, r*len(self.loader), 'repeat %d -- Mean Loss: %.3f | Last Loss: %.3f | Acc: %.3f%%'
                             % (r, np.mean(losses), losses[-1], np.mean(accuracies)))

        return np.asarray(losses), np.mean(accuracies)
    

    def get_sample(self):
        iterator = iter(self.loader)
        inputs, _ = iterator.next()
        return inputs[0:1,...].to(self.device)
    
        
    def run(self, optimizer=None, manipulator=identity, mask=None):
        if optimizer:
            self.network.train()
            return self._pass(optimizer, manipulator=manipulator, mask=mask)
        else:
            self.network.eval()
            with torch.no_grad():
                return self._pass(manipulator=manipulator, mask=mask)

            
    def get_predictions(self, manipulator=identity):
        ''' Returns predictions and targets '''
        preds, gts, = [], []
        for batch_idx, (inputs, targets) in enumerate(self.loader):
            targets = manipulator(targets)
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.network(inputs)
                
            gts.append(targets.cpu().data.numpy())
            preds.append(outputs.cpu().data.numpy().argmax(1))
            
        return np.concatenate(gts), np.concatenate(preds)

            
    def get_function(self, forward='selected'):
        ''' Collect function (features) from the self.network.module.forward_features() routine '''
        features = []
        for batch_idx, (inputs, targets) in enumerate(self.loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.network(inputs)

            if forward=='selected':
                features.append([f.cpu().data.numpy().astype(np.float16) for f in self.network.module.forward_features(inputs)])
            elif forward=='parametric':
                features.append([f.cpu().data.numpy().astype(np.float16) for f in self.network.module.forward_param_features(inputs)])
                
            progress_bar(batch_idx, len(self.loader))

        return [np.concatenate(list(zip(*features))[i]) for i in range(len(features[0]))]

    
    def get_structure(self):
        ''' Collect structure (weights) from the self.network.module.forward_weights() routine '''
        # modified #
        ## NOTICE: only weights are maintained and combined into two dimensions, biases are ignored
        weights = []
        [print("we get data type is {}, size is {}".format(type(f.data),f.size())) for f in self.network.parameters()]
        for index, var in enumerate(self.network.parameters()):
            if index % 2 == 0:
                f = var.cpu().data.numpy().astype(np.float16) # var as Variable, type(var.data) is Tensor, should be transformed from cuda to cpu(),with type float16
                weight = np.reshape(f, (f.shape[0], np.prod(f.shape[1:])))
                print("weight size ==== ", weight.shape)
                weights.append(weight)
       
        return weights
