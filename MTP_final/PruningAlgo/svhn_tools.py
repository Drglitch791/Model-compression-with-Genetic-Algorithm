import torch
from torch import nn

import math, gc, copy
from tqdm import tqdm
import numpy as np

class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        if '34.weight' in self.features.state_dict().keys():
            inshape = self.features.state_dict()['34.weight'].shape[0]
        else:
            inshape = self.features.state_dict()['features.34.weight'].shape[0]
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(inshape, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
        ## Initialize weights (kernels with normal randoms, bias with 0s)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def test(dataloader, models):
    # print("preparing models--")
    device = "cuda:0"
    size = len(dataloader.dataset)
    
    ## model_dicts is a list that stores
    ## all the models in same order as in list models
    model_dicts = []
    for model in models:
        model_dicts.append({
            'model':model,
            'correct':0,
            'pred':0
        })
    model_dicts
    ## test all the models
    # pbar = tqdm(total=len(dataloader))
    with torch.no_grad():
        iteration = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            for model_dict in model_dicts:
                model_dict['model'] = torch.nn.DataParallel(model_dict['model'], device_ids=[0])
                model_dict['model'].eval()
                model_dict['pred'] = model_dict['model'](x)
                model_dict['correct'] += (model_dict['pred'].argmax(1) == y).type(torch.float).sum().item()
                iteration += 1
            # pbar.update(1)
    # pbar.close()
    del x, y

    ## ret list will store accuracies of all the models
    ret_list = []
    for model_dict in model_dicts:
        del model_dict['model']
        model_dict['correct'] /= size
        ret_list.append(model_dict['correct']*100)
    gc.collect()
    torch.cuda.empty_cache()

    ## return the accuracies of all the models in same order
    return list(ret_list)

def get_model_filters_config(individual):
    filters_state = [
        np.count_nonzero(individual[0:64]==1),
        np.count_nonzero(individual[64:128]==1),
        'M',
        np.count_nonzero(individual[128:256]==1),
        np.count_nonzero(individual[256:384]==1),
        'M',
        np.count_nonzero(individual[384:640]==1),
        np.count_nonzero(individual[640:896]==1),
        np.count_nonzero(individual[896:1152]==1),
        np.count_nonzero(individual[1152:1408]==1),
        'M',
        np.count_nonzero(individual[1408:1920]==1),
        np.count_nonzero(individual[1920:2432]==1),
        np.count_nonzero(individual[2432:2944]==1),
        np.count_nonzero(individual[2944:3456]==1),
        'M',
        np.count_nonzero(individual[3456:3968]==1),
        np.count_nonzero(individual[3968:4480]==1),
        np.count_nonzero(individual[4480:4992]==1),
        np.count_nonzero(individual[4992:5504]==1),
        'M'
    ]
    return filters_state


def config_model_proper(individual, base_model):
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M']
    filters_state = {
        'features.0.weight': individual[0:64],
        'features.2.weight': individual[64:128],
        'features.5.weight': individual[128:256],
        'features.7.weight': individual[256:384],
        'features.10.weight': individual[384:640],
        'features.12.weight': individual[640:896],
        'features.14.weight': individual[896:1152],
        'features.16.weight': individual[1152:1408],
        'features.19.weight': individual[1408:1920],
        'features.21.weight': individual[1920:2432],
        'features.23.weight': individual[2432:2944],
        'features.25.weight': individual[2944:3456],
        'features.28.weight': individual[3456:3968],
        'features.30.weight': individual[3968:4480],
        'features.32.weight': individual[4480:4992],
        'features.34.weight': individual[4992:5504],
    }
    temp_model = VGG(make_layers(cfg))
    temp_model = temp_model.to('cuda')
    temp_model.load_state_dict(base_model.state_dict(), strict=True)
    model_state_dict = temp_model.state_dict()
    
    ## configure each layers
    ## (1)
    for filter_idx, filter in enumerate(filters_state['features.0.weight']):
        if filter == 0:
            model_state_dict['features.0.weight'][filter_idx,:,:,:] = 0
            model_state_dict['features.0.bias'][filter_idx] = 0

            model_state_dict['features.2.weight'][:,filter_idx,:,:] = 0
    ## (2)
    for filter_idx, filter in enumerate(filters_state['features.2.weight']):
        if filter == 0:
            model_state_dict['features.2.weight'][filter_idx,:,:,:] = 0
            model_state_dict['features.2.bias'][filter_idx] = 0

            model_state_dict['features.5.weight'][:,filter_idx,:,:] = 0
    ## (3)
    for filter_idx, filter in enumerate(filters_state['features.5.weight']):
        if filter == 0:
            model_state_dict['features.5.weight'][filter_idx,:,:,:] = 0
            model_state_dict['features.5.bias'][filter_idx] = 0

            model_state_dict['features.7.weight'][:,filter_idx,:,:] = 0
    ## (4)
    for filter_idx, filter in enumerate(filters_state['features.7.weight']):
        if filter == 0:
            model_state_dict['features.7.weight'][filter_idx,:,:,:] = 0
            model_state_dict['features.7.bias'][filter_idx] = 0

            model_state_dict['features.10.weight'][:,filter_idx,:,:] = 0
    ## (5)
    for filter_idx, filter in enumerate(filters_state['features.10.weight']):
        if filter == 0:
            model_state_dict['features.10.weight'][filter_idx,:,:,:] = 0
            model_state_dict['features.10.bias'][filter_idx] = 0

            model_state_dict['features.12.weight'][:,filter_idx,:,:] = 0
    ## (6)
    for filter_idx, filter in enumerate(filters_state['features.12.weight']):
        if filter == 0:
            model_state_dict['features.12.weight'][filter_idx,:,:,:] = 0
            model_state_dict['features.12.bias'][filter_idx] = 0

            model_state_dict['features.14.weight'][:,filter_idx,:,:] = 0
    ## (7)
    for filter_idx, filter in enumerate(filters_state['features.14.weight']):
        if filter == 0:
            model_state_dict['features.14.weight'][filter_idx,:,:,:] = 0
            model_state_dict['features.14.bias'][filter_idx] = 0

            model_state_dict['features.16.weight'][:,filter_idx,:,:] = 0
    ## (8)
    for filter_idx, filter in enumerate(filters_state['features.16.weight']):
        if filter == 0:
            model_state_dict['features.16.weight'][filter_idx,:,:,:] = 0
            model_state_dict['features.16.bias'][filter_idx] = 0

            model_state_dict['features.19.weight'][:,filter_idx,:,:] = 0
    ## (9)
    for filter_idx, filter in enumerate(filters_state['features.19.weight']):
        if filter == 0:
            model_state_dict['features.19.weight'][filter_idx,:,:,:] = 0
            model_state_dict['features.19.bias'][filter_idx] = 0

            model_state_dict['features.21.weight'][:,filter_idx,:,:] = 0
    ## (10)
    for filter_idx, filter in enumerate(filters_state['features.21.weight']):
        if filter == 0:
            model_state_dict['features.21.weight'][filter_idx,:,:,:] = 0
            model_state_dict['features.21.bias'][filter_idx] = 0

            model_state_dict['features.23.weight'][:,filter_idx,:,:] = 0
    ## (11)
    for filter_idx, filter in enumerate(filters_state['features.23.weight']):
        if filter == 0:
            model_state_dict['features.23.weight'][filter_idx,:,:,:] = 0
            model_state_dict['features.23.bias'][filter_idx] = 0

            model_state_dict['features.25.weight'][:,filter_idx,:,:] = 0
    ## (12)
    for filter_idx, filter in enumerate(filters_state['features.25.weight']):
        if filter == 0:
            model_state_dict['features.25.weight'][filter_idx,:,:,:] = 0
            model_state_dict['features.25.bias'][filter_idx] = 0

            model_state_dict['features.28.weight'][:,filter_idx,:,:] = 0

    ## (13)
    for filter_idx, filter in enumerate(filters_state['features.28.weight']):
        if filter == 0:
            model_state_dict['features.28.weight'][filter_idx,:,:,:] = 0
            model_state_dict['features.28.bias'][filter_idx] = 0

            model_state_dict['features.30.weight'][:,filter_idx,:,:] = 0
    
    ## (14)
    for filter_idx, filter in enumerate(filters_state['features.30.weight']):
        if filter == 0:
            model_state_dict['features.30.weight'][filter_idx,:,:,:] = 0
            model_state_dict['features.30.bias'][filter_idx] = 0

            model_state_dict['features.32.weight'][:,filter_idx,:,:] = 0

    ## (15)
    for filter_idx, filter in enumerate(filters_state['features.32.weight']):
        if filter == 0:
            model_state_dict['features.32.weight'][filter_idx,:,:,:] = 0
            model_state_dict['features.32.bias'][filter_idx] = 0

            model_state_dict['features.34.weight'][:,filter_idx,:,:] = 0
    
    ## (16) last conv layer is left unmodified to leave it stay compatible with fc layers
    ## jk. we will modify the 13th layer too. We will squeeze the network upto its bones.
    for filter_idx, filter in enumerate(filters_state['features.34.weight']):
        if filter == 0:
            model_state_dict['features.34.weight'][filter_idx,:,:,:] = 0
            model_state_dict['features.34.bias'][filter_idx] = 0

            model_state_dict['classifier.1.weight'][:,filter_idx] = 0 
            model_state_dict['classifier.1.bias'][filter_idx] = 0

    return temp_model

def calculate_params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    return pytorch_total_params

def make_model(individual, base_model):
    filters_state = {
        'features.0.weight': individual[0:64],
        'features.2.weight': individual[64:128],
        'features.5.weight': individual[128:256],
        'features.7.weight': individual[256:384],
        'features.10.weight': individual[384:640],
        'features.12.weight': individual[640:896],
        'features.14.weight': individual[896:1152],
        'features.16.weight': individual[1152:1408],
        'features.19.weight': individual[1408:1920],
        'features.21.weight': individual[1920:2432],
        'features.23.weight': individual[2432:2944],
        'features.25.weight': individual[2944:3456],
        'features.28.weight': individual[3456:3968],
        'features.30.weight': individual[3968:4480],
        'features.32.weight': individual[4480:4992],
        'features.34.weight': individual[4992:5504],
    }

    cfg_new = [
        np.count_nonzero(filters_state['features.0.weight']==1),
        np.count_nonzero(filters_state['features.2.weight']==1),
        'M',
        np.count_nonzero(filters_state['features.5.weight']==1),
        np.count_nonzero(filters_state['features.7.weight']==1),
        'M',
        np.count_nonzero(filters_state['features.10.weight']==1),
        np.count_nonzero(filters_state['features.12.weight']==1),
        np.count_nonzero(filters_state['features.14.weight']==1),
        np.count_nonzero(filters_state['features.16.weight']==1),
        'M',
        np.count_nonzero(filters_state['features.19.weight']==1),
        np.count_nonzero(filters_state['features.21.weight']==1),
        np.count_nonzero(filters_state['features.23.weight']==1),
        np.count_nonzero(filters_state['features.25.weight']==1),
        'M',
        np.count_nonzero(filters_state['features.28.weight']==1),
        np.count_nonzero(filters_state['features.30.weight']==1),
        np.count_nonzero(filters_state['features.32.weight']==1),
        np.count_nonzero(filters_state['features.34.weight']==1),
        'M'
    ]

    temp_model = VGG(make_layers(cfg_new))
    temp_model = temp_model.to('cuda')

    model_state_dict = copy.deepcopy(base_model.state_dict())
    temp_state_dict = copy.deepcopy(temp_model.state_dict())

    ## (1)
    fil = 0
    for filter_idx, filter in enumerate(filters_state['features.0.weight']):
        if filter == 1:
            temp_state_dict['features.0.weight'][fil,:,:,:] = model_state_dict['features.0.weight'][filter_idx,:,:,:]
            temp_state_dict['features.0.bias'][fil] = model_state_dict['features.0.bias'][filter_idx]

            temp_state_dict['features.2.weight'][:,fil,:,:] = model_state_dict['features.2.weight'][filters_state['features.2.weight']==1,filter_idx,:,:]
            fil += 1
    ## (2)
    fil = 0
    for filter_idx, filter in enumerate(filters_state['features.2.weight']):
        if filter == 1:
            temp_state_dict['features.2.weight'][fil,:,:,:] = model_state_dict['features.2.weight'][filter_idx,filters_state['features.0.weight']==1,:,:]
            temp_state_dict['features.2.bias'][fil] = model_state_dict['features.2.bias'][filter_idx]

            temp_state_dict['features.5.weight'][:,fil,:,:] = model_state_dict['features.5.weight'][filters_state['features.5.weight']==1,filter_idx,:,:]
            fil += 1
    ## (3)
    fil = 0
    for filter_idx, filter in enumerate(filters_state['features.5.weight']):
        if filter == 1:
            temp_state_dict['features.5.weight'][fil,:,:,:] = model_state_dict['features.5.weight'][filter_idx,filters_state['features.2.weight']==1,:,:]
            temp_state_dict['features.5.bias'][fil] = model_state_dict['features.5.bias'][filter_idx]

            temp_state_dict['features.7.weight'][:,fil,:,:] = model_state_dict['features.7.weight'][filters_state['features.7.weight']==1,filter_idx,:,:]
            fil += 1
    ## (4)
    fil = 0
    for filter_idx, filter in enumerate(filters_state['features.7.weight']):
        if filter == 1:
            temp_state_dict['features.7.weight'][fil,:,:,:] = model_state_dict['features.7.weight'][filter_idx,filters_state['features.5.weight']==1,:,:]
            temp_state_dict['features.7.bias'][fil] = model_state_dict['features.7.bias'][filter_idx]

            temp_state_dict['features.10.weight'][:,fil,:,:] = model_state_dict['features.10.weight'][filters_state['features.10.weight']==1,filter_idx,:,:]
            fil += 1
    ## (5)
    fil = 0
    for filter_idx, filter in enumerate(filters_state['features.10.weight']):
        if filter == 1:
            temp_state_dict['features.10.weight'][fil,:,:,:] = model_state_dict['features.10.weight'][filter_idx,filters_state['features.7.weight']==1,:,:]
            temp_state_dict['features.10.bias'][fil] = model_state_dict['features.10.bias'][filter_idx]

            temp_state_dict['features.12.weight'][:,fil,:,:] = model_state_dict['features.12.weight'][filters_state['features.12.weight']==1,filter_idx,:,:]
            fil += 1
    ## (6)
    fil = 0
    for filter_idx, filter in enumerate(filters_state['features.12.weight']):
        if filter == 1:
            temp_state_dict['features.12.weight'][fil,:,:,:] = model_state_dict['features.12.weight'][filter_idx,filters_state['features.10.weight']==1,:,:]
            temp_state_dict['features.12.bias'][fil] = model_state_dict['features.12.bias'][filter_idx]

            temp_state_dict['features.14.weight'][:,fil,:,:] = model_state_dict['features.14.weight'][filters_state['features.14.weight']==1,filter_idx,:,:]
            fil += 1
    ## (7)
    fil = 0
    for filter_idx, filter in enumerate(filters_state['features.14.weight']):
        if filter == 1:
            temp_state_dict['features.14.weight'][fil,:,:,:] = model_state_dict['features.14.weight'][filter_idx,filters_state['features.12.weight']==1,:,:]
            temp_state_dict['features.14.bias'][fil] = model_state_dict['features.14.bias'][filter_idx]

            temp_state_dict['features.16.weight'][:,fil,:,:] = model_state_dict['features.16.weight'][filters_state['features.16.weight']==1,filter_idx,:,:]
            fil += 1
    ## (8)
    fil = 0
    for filter_idx, filter in enumerate(filters_state['features.16.weight']):
        if filter == 1:
            temp_state_dict['features.16.weight'][fil,:,:,:] = model_state_dict['features.16.weight'][filter_idx,filters_state['features.14.weight']==1,:,:]
            temp_state_dict['features.16.bias'][fil] = model_state_dict['features.16.bias'][filter_idx]

            temp_state_dict['features.19.weight'][:,fil,:,:] = model_state_dict['features.19.weight'][filters_state['features.19.weight']==1,filter_idx,:,:]
            fil += 1
    ## (9)
    fil = 0
    for filter_idx, filter in enumerate(filters_state['features.19.weight']):
        if filter == 1:
            temp_state_dict['features.19.weight'][fil,:,:,:] = model_state_dict['features.19.weight'][filter_idx,filters_state['features.16.weight']==1,:,:]
            temp_state_dict['features.19.bias'][fil] = model_state_dict['features.19.bias'][filter_idx]

            temp_state_dict['features.21.weight'][:,fil,:,:] = model_state_dict['features.21.weight'][filters_state['features.21.weight']==1,filter_idx,:,:]
            fil += 1
    ## (10)
    fil = 0
    for filter_idx, filter in enumerate(filters_state['features.21.weight']):
        if filter == 1:
            temp_state_dict['features.21.weight'][fil,:,:,:] = model_state_dict['features.21.weight'][filter_idx,filters_state['features.19.weight']==1,:,:]
            temp_state_dict['features.21.bias'][fil] = model_state_dict['features.21.bias'][filter_idx]

            temp_state_dict['features.23.weight'][:,fil,:,:] = model_state_dict['features.23.weight'][filters_state['features.23.weight']==1,filter_idx,:,:]
            fil += 1
    ## (11)
    fil = 0
    for filter_idx, filter in enumerate(filters_state['features.23.weight']):
        if filter == 1:
            temp_state_dict['features.23.weight'][fil,:,:,:] = model_state_dict['features.23.weight'][filter_idx,filters_state['features.21.weight']==1,:,:]
            temp_state_dict['features.23.bias'][fil] = model_state_dict['features.23.bias'][filter_idx]

            temp_state_dict['features.25.weight'][:,fil,:,:] = model_state_dict['features.25.weight'][filters_state['features.25.weight']==1,filter_idx,:,:]
            fil += 1
    ## (12)
    fil = 0
    for filter_idx, filter in enumerate(filters_state['features.25.weight']):
        if filter == 1:
            temp_state_dict['features.25.weight'][fil,:,:,:] = model_state_dict['features.25.weight'][filter_idx,filters_state['features.23.weight']==1,:,:]
            temp_state_dict['features.25.bias'][fil] = model_state_dict['features.25.bias'][filter_idx]

            temp_state_dict['features.28.weight'][:,fil,:,:] = model_state_dict['features.28.weight'][filters_state['features.28.weight']==1,filter_idx,:,:]
            fil += 1
    
    ## (13)
    fil = 0
    for filter_idx, filter in enumerate(filters_state['features.28.weight']):
        if filter == 1:
            temp_state_dict['features.28.weight'][fil,:,:,:] = model_state_dict['features.28.weight'][filter_idx,filters_state['features.25.weight']==1,:,:]
            temp_state_dict['features.28.bias'][fil] = model_state_dict['features.28.bias'][filter_idx]

            temp_state_dict['features.30.weight'][:,fil,:,:] = model_state_dict['features.30.weight'][filters_state['features.30.weight']==1,filter_idx,:,:]
            fil += 1
    
    ## (14)
    fil = 0
    for filter_idx, filter in enumerate(filters_state['features.30.weight']):
        if filter == 1:
            temp_state_dict['features.30.weight'][fil,:,:,:] = model_state_dict['features.30.weight'][filter_idx,filters_state['features.28.weight']==1,:,:]
            temp_state_dict['features.30.bias'][fil] = model_state_dict['features.30.bias'][filter_idx]

            temp_state_dict['features.32.weight'][:,fil,:,:] = model_state_dict['features.32.weight'][filters_state['features.32.weight']==1,filter_idx,:,:]
            fil += 1
    
    ## (15)
    fil = 0
    for filter_idx, filter in enumerate(filters_state['features.32.weight']):
        if filter == 1:
            temp_state_dict['features.32.weight'][fil,:,:,:] = model_state_dict['features.32.weight'][filter_idx,filters_state['features.30.weight']==1,:,:]
            temp_state_dict['features.32.bias'][fil] = model_state_dict['features.32.bias'][filter_idx]

            temp_state_dict['features.34.weight'][:,fil,:,:] = model_state_dict['features.34.weight'][filters_state['features.34.weight']==1,filter_idx,:,:]
            fil += 1

    ## (13)
    fil = 0
    for filter_idx, filter in enumerate(filters_state['features.34.weight']):
        if filter == 1:
            temp_state_dict['features.34.weight'][fil,:,:,:] = model_state_dict['features.34.weight'][filter_idx,filters_state['features.32.weight']==1,:,:]
            temp_state_dict['features.34.bias'][fil] = model_state_dict['features.34.bias'][filter_idx]
            
            temp_state_dict['classifier.1.weight'][:,fil] = model_state_dict['classifier.1.weight'][:,filter_idx]
            temp_state_dict['classifier.1.bias'][fil] = model_state_dict['classifier.1.bias'][filter_idx]
            fil += 1

    # temp_model.classifier = copy.deepcopy(base_model.classifier)
    temp_state_dict['classifier.4.weight'] = model_state_dict['classifier.4.weight']
    temp_state_dict['classifier.4.bias'] = model_state_dict['classifier.4.bias']
    temp_state_dict['classifier.6.weight'] = model_state_dict['classifier.6.weight']
    temp_state_dict['classifier.6.bias'] = model_state_dict['classifier.6.bias']

    temp_model.load_state_dict(temp_state_dict)

    return temp_model