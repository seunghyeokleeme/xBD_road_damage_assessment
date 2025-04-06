import os
import torch

def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))

def load(ckpt_dir, net, optim, device, weights_only=True):
    if not os.path.exists(ckpt_dir):
        epoch  = 0
        return net, optim, epoch
    ckpt_list = os.listdir(ckpt_dir)
    ckpt_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_list[-1]), map_location=device, weights_only=weights_only)

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_list[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch