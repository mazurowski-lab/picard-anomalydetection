##############################
# code for predicting anomaly detection heatmaps of images
# using pluralistic image completion.
##############################

### IMPORTS
import os
from argparse import ArgumentParser

from utils import *
from modules import *
from heatmapping import *
from eval import *

# torch
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

# other libs
from datetime import datetime
import random
from random import sample
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from time import time

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml',
                    help="configuration")
parser.add_argument('--seed', type=int, help='manual seed', default=1337,
                    help="manual random seed")
parser.add_argument('--checkpoint_dir', type=str,
                    help="path to saved inpainter model checkpoint directory")
parser.add_argument('--checkpoint_iter', type=int,
                    help="iteration number of saved model checkpoint")

def main():
    args = parser.parse_args()
    config = get_config(args.config)

    ############################################################
    ### (1) GPU setup
    ############################################################
    cuda = config['cuda']
    device_check = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on {}'.format(device_check))

    # set which devices CUDA sees
    device_ids = config['gpu_ids'] # indices of devices for models, data and otherwise
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)

    # all devices are then indexed from this set
    model_device = 0

    # set random seed
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


    # logger
    log_dir = 'test_logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = Logger('test', log_dir, heatmap_metrics)
    dt_now = datetime.now()

    ############################################################
    ### (2) model setup
    ############################################################
    # choose model and dataset we'll be working with
    model_type = 'dropout'
    dataset_name = config['dataset_name']

    PIL_img_format = 'L' if (config["test"]["patch_shape"][-1] == 1) else 'RGB'
    completion_img_size = config["test"]["patch_shape"][0]
    # ^ side size of image to be completed (may be just part of larger heatmap image)

    # load utils
    normalize_img = load_img_normalizer(model_type)

    # hyperparameter and checkpoint setup 
    hyperparams = {
        'dropout' : {
            'p_dropout' : config["test"]["droprate"]
        },
    }

    checkpoints = {
        'dropout' : {
            'gen' : args.checkpoint_dir,
            'dis' : args.checkpoint_dir,
            'iter' : args.checkpoint_iter
        },
    }

    # load inpainter and completion feature extractor
    inpainter = load_multi_inpainter(
        model_type, 
        checkpoints[model_type], 
        hyperparams[model_type], 
        device_ids,
        dropoutmodel_config=args.config
    )

    feature_extractor = load_inpainting_feature_extractor(
        model_type, 
        checkpoints[model_type], 
        hyperparams[model_type], 
        device_ids,
        dropoutmodel_config=args.config
    )

    # heatmapping settings
    # visualization and analysis settings
    save_heatmap_data = config["test"]["save_heatmap_data"]
    save_heatmap_plots = config["test"]["save_heatmap_plots"] 

    save_progressive_heatmap = config["test"]["save_progressive_heatmap"]
    log_compute_times = config["test"]["log_compute_times"]

    # heatmapping parameters
    mask_size = config["test"]["mask_shape"][0]
    window_size = config["test"]["patch_shape"][0]
    window_stride = config["test"]["patch_stride"]
    heatmap_M_inpaint = config["test"]["heatmap_M_inpaint"]
    heatmap_metrics = config["test"]["heatmap_metrics"]
    parallel_batchsize = config["test"]["parallel_batchsize"]

    # misc scoring settings
    only_check_nonblack_pixels = config["test"]["only_check_nonblack_pixels"]


    ############################################################
    ### (3) heatmap generation
    ############################################################

    test_data_fnames = [os.path.join(config["test_data_path"], f) for f in os.listdir(config["test_data_path"])]

    with torch.no_grad():
        for test_data in test_data_fnames:
            # 1) load image
            img = pil_loader(test_data, img_format=PIL_img_format)
            img = transforms.ToTensor()(img)
            # don't normalize at the image level: will normalize at the patch level
            img = img.unsqueeze(dim=0)
            
            # plot bbox on img
            img = img.cpu()
            print(test_data)
            show_images(img, custom_figsize=(10, 14))
            img = img.cuda()
            
            ignore_mask = None
            if only_check_nonblack_pixels:
                print('ONLY CHECKING NONBLACK PIXELS')
                # mask of size image; True where there are pixels that 
                # we don't want to include in evaluation
                ignore_mask = (normalize_img(img) == -1.).cpu()

            # 2) generate heatmaps
            tin = time()
            heatmaps = generate_anomaly_heatmap_slidingwindow_PARALLEL(
                img, 
                inpainter,
                feature_extractor,
                metrics=heatmap_metrics,                                                   
                mask_size=mask_size,
                window_size=window_size,
                window_stride=window_stride,            
                M_inpaint=heatmap_M_inpaint,
                heatmap_batch_size=parallel_batchsize,
                heatmap_type='nonaveraged',
                img_normalizer = normalize_img,
                save_progressive_heatmap = save_progressive_heatmap
            )
            tout = time()
            
            print('time to create heatmap = {} sec'.format(tout - tin))
            
            # plot and save heatmap data and images
            for heatmap_metric in heatmap_metrics:
                # create dirs
                savedir = os.path.join('heatmaps', dataset_name, model_type, dt_now.strftime("%m-%d-%Y_%H:%M:%S"))
                savedir_maps = os.path.join(savedir, 'data')
                savedir_plots = os.path.join(savedir, 'plots')
                for path in [savedir_maps, savedir_plots]:
                    if not os.path.exists(path):
                        os.makedirs(path)
                        
                # save heatmap data
                filename = test_data
                filename = filename.replace('.png', '')
                filename = filename.split('/')
                filename = filename[-1]
                filename += '_{}_{}_{}_{}_{}_{}.pt'.format(heatmap_metric,
                                                    heatmap_M_inpaint, 
                                                    hyperparams['dropout']['p_dropout'],
                                                    mask_size,
                                                    window_size,
                                                    window_stride,
                                                )
                filename_map = os.path.join(savedir_maps, filename)
                if save_heatmap_data:
                    torch.save(heatmaps[heatmap_metric], filename_map)

                # plot heatmap
                fig, ax = plt.subplots(figsize=(10, 14))
                im = ax.imshow((heatmaps[heatmap_metric]).cpu(), cmap=plt.cm.hot, interpolation='none') 
                cbar = fig.colorbar(im, extend='max')
                title = 'anomaly metric: {}\nM={}, p={}, mask size = {}\nwindow size = {}, window stride = {}'.format(
                                                    heatmap_metric,
                                                    heatmap_M_inpaint, 
                                                    hyperparams['dropout']['p_dropout'],
                                                    mask_size,
                                                    window_size,
                                                    window_stride,
                )
                plt.title(title, fontsize=20)
                
                # save heatmap plot 'data_new/test/cancer/val_DBT-P01700_DBT-S01353_lmlo_Cancer_0.png'
                filename_img = filename.replace('.pt', '.png')
                filename_img = os.path.join(savedir_plots, filename_img)
                if save_heatmap_plots:
                    plt.savefig(fname=filename_img, bbox_inches = 'tight')
                plt.show()
            
            # log heatmaps on image
            log_hyperparams = [window_stride, window_size, mask_size, 
                            heatmap_M_inpaint, parallel_batchsize, 'nonaveraged', 
                            hyperparams['dropout']['p_dropout']]
            if log_compute_times:
                logger.write_msg('heatmap compute time on {} GPUs = {}\n'.format(len(device_ids), tout-tin))

if __name__ == '__main__':
    main()