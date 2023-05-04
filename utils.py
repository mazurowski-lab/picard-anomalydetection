##############################
# misc. utilities
##############################
import os
import yaml
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import shutil
from datetime import datetime

# logging
class Logger():
    def __init__(self, mode, log_dir, all_heatmap_metrics=None, heatmap_validation_metric=None, search_space=None):
        assert mode in ['hyperopt', 'test', 'custom']
        self.mode = mode
        self.all_heatmap_metrics = all_heatmap_metrics
        
        # create log file
        now = datetime.now()
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        logfname = 'log_{}.txt'.format(now.strftime("%m-%d-%Y_%H:%M:%S"))
        self.logfname = os.path.join(log_dir, logfname)
        
        with open(self.logfname, 'w') as fp: # create file
            pass
        
        # log intro message
        start_msg = 'beginning {} on {}.\n'.format(self.mode, now.strftime("%m/%d/%Y, %H:%M:%S"))
        
        if mode == 'hyperopt':
            self.heatmap_validation_metric = heatmap_validation_metric
            
            start_msg += 'search space:\n' 
            for s in search_space:
                start_msg += '{}: {}\n'.format(s.name, str(s))
            start_msg += '\n'
            start_msg += '--------------------------\n'
            start_msg += 'datetime | primary score | secondary score(s) | hyperparams\n'
            
        elif mode == 'test':
            start_msg += '--------------------------\n'
            start_msg += 'datetime | img_fname | heatmap metric scores {} | hyperparams\n'.format(self.all_heatmap_metrics)
            
        elif mode == 'custom':
            start_msg += '--------------------------\n'
            start_msg += 'custom log.\n'
        
        self.write_msg(start_msg)
        print(start_msg)
        
    def write_msg(self, msg):
        log_f = open(self.logfname, 'a')
        log_f.write(msg)
        log_f.close()
        
        return
        
    def log_run(self, hyperparams, scores, img_fname=None):
        now = datetime.now()
        run_msg = '{} '.format(now.strftime("%m/%d/%Y, %H:%M:%S"))
        if self.mode == 'hyper_opt':
            run_msg += '{} '.format(scores[self.heatmap_validation_metric])
            for m in self.all_heatmap_metrics:
                if m != self.heatmap_validation_metric:
                    run_msg += '{} '.format(scores[m])
        elif self.mode == 'test':
            assert img_fname is not None
            run_msg += '{} '.format(img_fname)
            for m in self.all_heatmap_metrics:
                run_msg += '{} '.format(scores[m])
        for p in hyperparams:
            run_msg += '{} '.format(p)
        run_msg += '\n'
        self.write_msg(run_msg)
        print(run_msg)
        return
        



# utils
def pil_loader(path, img_format):
    # load PIL img from file
    with open(path, 'rb') as f:
        img = Image.open(f)
        img.load()
        return img.convert(img_format)

# get configs
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)
    
# image tensor normalization (model dependent)
def yuetal18_normalizer(x): 
    # non in-place version of normalization
    # for contextual attnetion inpainter
    # (normalize from [0, 1] to [-1, 1]
    x_norm = (2.* x) - 1. 
    return x_norm

def HFPIC_normalizer(x):
    # HFPIC receives images as-is/does normalization internally
    return x

def load_img_normalizer(model_type):
    normalizer = None
    if model_type == 'dropout':
        normalizer = yuetal18_normalizer
    elif model_type == 'HFPIC':
        normalizer = HFPIC_normalizer
        
    return normalizer

def show_images(imgs, single_img_size=6, custom_figsize=None, bboxes=None):
    ncol = min(imgs.shape[0], 4)
    grid_img = torchvision.utils.make_grid(imgs, nrow=ncol, normalize=True)
    grid_img = grid_img.cpu()#.numpy()

    if not custom_figsize:
        custom_figsize = (single_img_size*ncol, single_img_size * (imgs.shape[0] // ncol))
    plt.figure(figsize=custom_figsize)
    plt.imshow(grid_img.permute(1, 2, 0))
    
    # plot bboxes
    if bboxes:
        for bbox in bboxes:
            plt.gca().add_patch(Rectangle((bbox[1], bbox[0]), bbox[3], bbox[2],linewidth=4,edgecolor='r',facecolor='none'))
    
    plt.show()
    return

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)