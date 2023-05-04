"""
Tools for using dropout on models
"""
import torch
import torch.nn as nn

def customize_dropout(net, config, verbose=False):
    # on a network that is either in train or eval mode, turn off all dropout layers
    # (put then in eval) EXCEPT for specified ones

    net.apply(apply_dropout_off) # turn off all dropout layers
    if verbose: print('turning off all dropouts...')
    
    for n, m in net.named_modules():
        # apply dropout to custom layers
        #print(n)
        if 'dropout' in n:
            assert type(m) == nn.Dropout2d or type(m) == nn.Dropout
            if 'coarse_generator' in n:
                for custom_name in config['netG']['custom_drop_layers_coarse']:
                    if custom_name in n:
                        if verbose: print('applying dropout only to {}'.format(n))
                        m.train()
            elif 'fine_generator' in n:
                for custom_name in config['netG']['custom_drop_layers_fine']:
                    if custom_name in n:
                        if verbose: print('applying dropout only to {}'.format(n))
                        m.train()
            

def apply_dropout_on(m, verbose=False): 
    # make it so that when network is being tested (on .eval()), dropout is still used
    if type(m) == nn.Dropout2d or type(m) == nn.Dropout:
        if verbose: print('turning Dropout module {} to train()...'.format(m))
        m.train()

def apply_dropout_off(m, verbose=False):
    if type(m) == nn.Dropout2d or type(m) == nn.Dropout:
        if verbose: print('turning Dropout module {} to eval()...'.format(m))
        m.eval()

def adjust_droprate(m, p_new):
    def apply_adjust(m):
        if type(m) == nn.Dropout2d or type(m) == nn.Dropout:
            m.p = p_new

    return apply_adjust

def remove_last_dropout(model):
    # turn last dropout layer to .eval for testing
    m = list(model.modules())[-1]
    assert type(m) == nn.Dropout2d or type(m) == nn.Dropout, 'last module of net is not a dropout'
    print('turning off last dropout layer in model.')
    m.eval()
