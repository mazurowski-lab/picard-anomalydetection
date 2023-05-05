##############################
# functions used to generate anomaly detection heatmaps of images.
# (see predict_heatmap.py for usage to actual create heatmaps)
##############################
import torch
import torch.utils.data as data
from torch.nn.functional import interpolate
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from utils import show_images

# heatmapping code: sliding surroundings/sliding window

# for parallel window evaluation (not implemented)
def get_2Dindices_from1D(obj, index): # used to convert a 1D index over all of the patches to the 2D image grid
    w_gridpoint_index = index % obj.num_gridpoints_w # which sliding window gridpoint along width are you currently on
    h_gridpoint_index = index // obj.num_gridpoints_w # same along height

    return w_gridpoint_index, h_gridpoint_index

class WindowDataset(data.Dataset):
    def __init__(self, img, window_size, window_stride, img_normalizer):
        super(WindowDataset, self).__init__()
        self.master_img = img
        img_height = img.shape[2]
        img_width = img.shape[3]

        # compute total number of sliding window iterations
        self.num_gridpoints_w = (img_width - window_size) // window_stride + 1
        self.num_gridpoints_h = (img_height - window_size) // window_stride + 1
        self.num_iter_slide = self.num_gridpoints_w * self.num_gridpoints_h
        
        self.window_size = window_size
        self.window_stride = window_stride
        self.normalize_img = img_normalizer

    
    def __getitem__(self, index):
        # index here is the index of which patch that we're looking at, out of all of the sliding window patches, ordered left->right, top->bottom 
        w_gridpoint_index, h_gridpoint_index = get_2Dindices_from1D(self, index)
        patch_center_coord_w = w_gridpoint_index * self.window_stride + self.window_size // 2 # coords of patch center
        patch_center_coord_h = h_gridpoint_index * self.window_stride + self.window_size // 2
        crop_top_coord = patch_center_coord_h - self.window_size // 2 # coords defining patch boundaries 
        crop_left_coord = patch_center_coord_w - self.window_size // 2 

        patch = self.master_img[:, :, 
                                crop_top_coord:crop_top_coord+self.window_size, 
                                crop_left_coord:crop_left_coord+self.window_size]
        patch = self.normalize_img(patch)
        
        gridpoint_indices = torch.IntTensor([w_gridpoint_index, h_gridpoint_index])

        return patch, gridpoint_indices 

    def __len__(self):
        return self.num_iter_slide
    
def generate_anomaly_heatmap_slidingwindow_PARALLEL(img, inpainter, feature_extractor, mask_size, window_size, window_stride, 
                                                    M_inpaint, metrics, heatmap_batch_size, heatmap_type, img_normalizer,
                                                    normalize_residuals=False, ignore_black_regions=False,
                                                   plot_progress=False, log_progress=True, save_progressive_heatmap=False):
    ''' 
    generate anomaly heatmap for torch img tensor
    # inputs:   image, 
                mask_size (assuming square mask),
                mask_stride (pixels),
                M_inpaint (int),
                metric (MSE, MCD), (list of metrics)
                heatmap_type (averaged or nonaveraged)
                
    # return: heatmap (H x W)
    '''
    # set up loading windows/patches of full image
    img_height = img.shape[2]
    img_width = img.shape[3]
    window_dataset = WindowDataset(img, window_size, window_stride, img_normalizer)
    window_loader = torch.utils.data.DataLoader(dataset=window_dataset,
                                                       batch_size=heatmap_batch_size)
    niter = len(window_dataset)
    
    inpainter_input_size = 256
    critic_input_size = 128
    
    # create mask (fixed since surroundings slide around)
    mask_single = torch.zeros((1, 1, window_size, window_size))
    mask_single = mask_single.cuda()
    mask_single[:, :, (window_size-mask_size)//2:(window_size+mask_size)//2, (window_size-mask_size)//2:(window_size+mask_size)//2] = 1.
    
    # for now, do averaged heatmap
    if heatmap_type == 'averaged':
        heatmap_unnormalized = {} # one heatmap for each metric
        for metric in metrics:
            heatmap_unnormalized[metric] = torch.zeros_like(img) # torch.full_like(img, float('nan'))
        heatmap_inclusion_counts = torch.zeros_like(img)
    elif heatmap_type == 'nonaveraged':
        heatmap_H = len(range(0, img_height-window_size, window_stride))
        heatmap_W = len(range(0, img_width-window_size, window_stride))
        heatmap_coarse = {}
        for metric in metrics:
            heatmap_coarse[metric] = torch.zeros((heatmap_H, heatmap_W))
    
    # anchors for windows
    print('{} heatmap iterations total.'.format(len(window_loader)))
    
    for batch_idx, (img_window, gridpoint_indices) in tqdm(enumerate(window_loader), desc='Heatmap generating (parallelized)', disable = not log_progress):
        if len(img_window.shape) > 4: # still have no idea how this dim is being added
            img_window = torch.squeeze(img_window, dim=1)
            
        # resize windows and masks if needed
        if window_size != inpainter_input_size:
            img_window = interpolate(img_window, size=inpainter_input_size)
            mask_single = interpolate(mask_single, size=inpainter_input_size)
            
        # window is already normalized to [-1 1] in WindowLoader
        
        # init scores
        
        # (possibly) skip window(s) if they have a black pixel
        # indices of which windows to inpaint
        inpaint_indices = range(heatmap_batch_size)
        if ignore_black_regions:
            lowest_pixels = torch.min(img_window.view(img_window.shape[0], -1), dim=1)[0]
            min_pix = -1.
            skip_check = lowest_pixels == -1. # true if we should skip inpainting the window
            if skip_check.sum().item() > 0: # only keep certain images 
                if skip_check.sum().item() == heatmap_batch_size:
                    # do no inpaintings at all/score nothing and skip to next iter
                    continue
                inpaint_indices = (skip_check == False).nonzero(as_tuple=True)[0]
                # choose which windows to still inpaint
                img_window = torch.index_select(img_window, 0, inpaint_indices.cuda())
            # remove windows to be skipped from batch, 
        
        # make mask match batch dim of input imgs
        try:
            mask = torch.cat(img_window.shape[0]*[mask_single])
        except RuntimeError:
            continue
            
        # do inpaintings for all inputs
        inpaintings = inpainter(img_window, mask, M_inpaint)
        # (N x M x C x H x W)^

        # compare inpaintings to ground truth with metric of choice
        # get completions only (no surroundings)
        
        # get everything of dim (NxMxCxHxW)
        completions = inpaintings[:, :, :, (window_size-mask_size)//2:(window_size+mask_size)//2, (window_size-mask_size)//2:(window_size+mask_size)//2]
        completion_ground_truth = img_window[:, :, (window_size-mask_size)//2:(window_size+mask_size)//2, (window_size-mask_size)//2:(window_size+mask_size)//2]
        
        
        # pretrained feature extractor expects input dimensionality of 128*128, so resize if not
        if mask_size != critic_input_size:
            completions = interpolate(completions, size=(completions.shape[2], critic_input_size, critic_input_size))
            # ^ because this is already N x M x C x H x W, and this -v- is N x C x H x W
            completion_ground_truth = interpolate(completion_ground_truth, size=critic_input_size)
            
        completion_ground_truth = completion_ground_truth.unsqueeze(dim=1) # add M dim -> (N x 1 x C x H x W)
        completion_ground_truth = torch.cat(M_inpaint * [completion_ground_truth], dim=1) #(N x M x C x H x W)
        # compute anomaly scores for nonskipped windows
        scores = {}#{metric : [] for metric in metrics}
        for metric in metrics:
            # compute score with given metric
            if metric == 'MCD_image':
                residuals = completion_ground_truth.cuda() - completions.cuda()
                if normalize_residuals:
                    raise NotImplementedError

                L2 = torch.norm(residuals, p=2, dim=(2,3,4)) # L2 distances from completions to g.t.
                score = torch.min(L2, dim=1)[0] # for each input, minimum L2 distance of its completions to itself
                scores[metric] = score
                
                
            elif metric == 'MCD_feature':
                # extract features from completion images
                # to do forward pass, need in shape (N*M x C x H x W)
                completions_all = completions.view(completions.shape[0]*completions.shape[1], *completions.shape[2:])
                completion_ground_truth_all = completion_ground_truth.view(completion_ground_truth.shape[0]*completion_ground_truth.shape[1], *completion_ground_truth.shape[2:])
                
                completion_features = feature_extractor(completions_all)
                completion_features = torch.flatten(completion_features, start_dim=1) # (N*M x D)
                
                completion_gt_features = feature_extractor(completion_ground_truth_all)
                completion_gt_features = torch.flatten(completion_gt_features, start_dim=1) # (N*M x D)
                
                # put back into more transparent shape N x M x D
                completion_features = completion_features.view(completions.shape[0], completions.shape[1], completion_features.shape[1])
                completion_gt_features = completion_gt_features.view(completion_ground_truth.shape[0], completion_ground_truth.shape[1], completion_gt_features.shape[1])

                residuals = completion_gt_features.cuda() - completion_features.cuda()
                if normalize_residuals:
                    raise NotImplementedError

                L2 = torch.norm(residuals, p=2, dim=(2)) # L2 distances from completions to g.t.
                score = torch.min(L2, dim=1)[0] # minimum L2 distance
                scores[metric] = score
                
                
                
            ## ADDITIONAL METRICS
            elif metric == 'MeanCD_image':
                residuals = completion_ground_truth.cuda() - completions.cuda()
                if normalize_residuals:
                    raise NotImplementedError

                L2 = torch.norm(residuals, p=2, dim=(2,3,4)) # L2 distances from completions to g.t.
                score = torch.mean(L2, dim=1) # for each input, L2 distance of its completions to itself
                scores[metric] = score
                
                
            elif metric == 'MeanCD_feature':
                # extract features from completion images
                # to do forward pass, need in shape (N*M x C x H x W)
                completions_all = completions.view(completions.shape[0]*completions.shape[1], *completions.shape[2:])
                completion_ground_truth_all = completion_ground_truth.view(completion_ground_truth.shape[0]*completion_ground_truth.shape[1], *completion_ground_truth.shape[2:])
                
                completion_features = feature_extractor(completions_all)
                completion_features = torch.flatten(completion_features, start_dim=1) # (N*M x D)
                
                completion_gt_features = feature_extractor(completion_ground_truth_all)
                completion_gt_features = torch.flatten(completion_gt_features, start_dim=1) # (N*M x D)
                
                # put back into more transparent shape N x M x D
                completion_features = completion_features.view(completions.shape[0], completions.shape[1], completion_features.shape[1])
                completion_gt_features = completion_gt_features.view(completion_ground_truth.shape[0], completion_ground_truth.shape[1], completion_gt_features.shape[1])

                residuals = completion_gt_features.cuda() - completion_features.cuda()
                if normalize_residuals:
                    raise NotImplementedError

                L2 = torch.norm(residuals, p=2, dim=(2)) # L2 distances from completions to g.t.
                score = torch.mean(L2, dim=1) # L2 distance
                scores[metric] = score
                
                
                
            elif metric == 'MedCD_image':
                residuals = completion_ground_truth.cuda() - completions.cuda()
                if normalize_residuals:
                    raise NotImplementedError

                L2 = torch.norm(residuals, p=2, dim=(2,3,4)) # L2 distances from completions to g.t.
                score = torch.median(L2, dim=1)[0] # for each input, L2 distance of its completions to itself
                scores[metric] = score
                
                
            elif metric == 'MedCD_feature':
                # extract features from completion images
                # to do forward pass, need in shape (N*M x C x H x W)
                completions_all = completions.view(completions.shape[0]*completions.shape[1], *completions.shape[2:])
                completion_ground_truth_all = completion_ground_truth.view(completion_ground_truth.shape[0]*completion_ground_truth.shape[1], *completion_ground_truth.shape[2:])
                
                completion_features = feature_extractor(completions_all)
                completion_features = torch.flatten(completion_features, start_dim=1) # (N*M x D)
                
                completion_gt_features = feature_extractor(completion_ground_truth_all)
                completion_gt_features = torch.flatten(completion_gt_features, start_dim=1) # (N*M x D)
                
                # put back into more transparent shape N x M x D
                completion_features = completion_features.view(completions.shape[0], completions.shape[1], completion_features.shape[1])
                completion_gt_features = completion_gt_features.view(completion_ground_truth.shape[0], completion_ground_truth.shape[1], completion_gt_features.shape[1])

                residuals = completion_gt_features.cuda() - completion_features.cuda()
                if normalize_residuals:
                    raise NotImplementedError

                L2 = torch.norm(residuals, p=2, dim=(2)) # L2 distances from completions to g.t.
                score = torch.median(L2, dim=1)[0] # L2 distance
                scores[metric] = score
                
        inpainted_idx = 0
        for window_idx, gridpoint_2d_index in enumerate(gridpoint_indices.tolist()):
            if window_idx in inpaint_indices: # only add score to heatmap for non-skipped windows
                w_gridpoint_index, h_gridpoint_index = tuple(gridpoint_2d_index)

                if heatmap_type == 'averaged':
                    window_center_coord_w = w_gridpoint_index * window_stride + window_size // 2 # coords of patch center
                    window_center_coord_h = h_gridpoint_index * window_stride + window_size // 2

                    h_anchor = window_center_coord_h - window_size // 2 # coords defining patch boundaries 
                    w_anchor = window_center_coord_w - window_size // 2 

                    heatmap_inclusion_counts[:, :, h_anchor:h_anchor+window_size, w_anchor:w_anchor+window_size] += 1
                    for metric in metrics:
                        window_score = scores[metric][inpainted_idx].item()
                        heatmap_unnormalized[metric][:, :, h_anchor:h_anchor+window_size, w_anchor:w_anchor+window_size] += window_score
                        
                elif heatmap_type == 'nonaveraged':
                    for metric in metrics:
                        window_score = scores[metric][inpainted_idx].item()
                        heatmap_coarse[metric][h_gridpoint_index, w_gridpoint_index] = window_score

                inpainted_idx += 1 

        
        print_iter_heatmaps = False # for debugging
        if print_iter_heatmaps:
            for heatmap_metric in metrics:
                if heatmap_type == 'averaged':
                    heatmap = torch.div(heatmap_unnormalized[metric], heatmap_inclusion_counts)
                    print(heatmap)
                elif heatmap_type == 'nonaveraged':
                    heatmap = heatmap_coarse[metric]
                print('{}:'.format(heatmap_metric))
                print(heatmap.shape)
                print(heatmap)
                
        if save_progressive_heatmap:
            save_prog_dir = '/workspace/heatmaps/tmp'
            os.system('rm -rf {}/*'.format(save_prog_dir))
            for heatmap_metric in metrics:
                fname = os.path.join(save_prog_dir, '{}_{}_{}.pt' .format(heatmap_metric, batch_idx, heatmap_type))
                if heatmap_type == 'nonaveraged':
                    torch.save(heatmap_coarse[heatmap_metric], fname)
                else:
                    raise NotImplementedError
                print('saving heatmap iter {}...'.format(batch_idx))
            
        # iteratively show heatmap (for testing)
        if plot_progress:
            for heatmap_metric in metrics:
                if heatmap_type == 'averaged':
                    heatmap = torch.div(heatmap_unnormalized[metric], heatmap_inclusion_counts)
                elif heatmap_type == 'nonaveraged':
                    heatmap = heatmap_coarse[metric]
                fig, ax = plt.subplots(figsize=(10, 14))
                im = ax.imshow(heatmap.cpu(), cmap=plt.cm.hot, interpolation='none') #, vmax=threshold)
                # ax.imshow(seg, alpha=0.5)
                cbar = fig.colorbar(im, extend='max')
                plt.show()
                
    # create final normalized heatmaps
    heatmaps = {}
    for metric in metrics:         
        if heatmap_type == 'averaged':
            heatmap = torch.div(heatmap_unnormalized[metric], heatmap_inclusion_counts)
        elif heatmap_type == 'nonaveraged':
            heatmap = heatmap_coarse[metric]
        
        # heatmap = torch.squeeze(heatmap)
        if heatmap_type == 'nonaveraged':
            # upsample to get same size as image
            heatmap = torch.unsqueeze(heatmap, dim=0)
            heatmap = torch.unsqueeze(heatmap, dim=0)
            heatmap = interpolate(heatmap, size=(img_height, img_width), mode='bicubic')
        heatmap = torch.squeeze(heatmap)                                    
        global_heatmap = heatmap.cpu()
    
        # post-process
        heatmap = torch.nan_to_num(heatmap) # fix nans to zeroes (non-classified pixels) (for non-included pixels)
        
        heatmaps[metric] = heatmap
    return heatmaps

def generate_anomaly_heatmap_slidingwindow(img, mask_size, window_size, window_stride, M_inpaint, metrics, img_normalizer, heatmap_type='averaged', normalize_residuals=False):
    ''' 
    generate anomaly heatmap for torch img tensor
    # inputs:   image, 
                mask_size (assuming square mask),
                mask_stride (pixels),
                M_inpaint (int),
                metric (MSE, MCD), (list of metrics)
                heatmap_type (averaged or nonaveraged)
                
    # return: heatmap (H x W)
    '''
    normalize_img = img_normalizer
    # set up loading windows/patches of full image
    img_height = img.shape[2]
    img_width = img.shape[3]
    
    # create mask (fixed since surroundings slide around)
    mask = torch.zeros((1, 1, window_size, window_size))
    mask = mask.cuda()
    mask[:, :, (window_size-mask_size)//2:(window_size+mask_size)//2, (window_size-mask_size)//2:(window_size+mask_size)//2] = 1.
    
    # for now, do averaged heatmap
    if heatmap_type == 'averaged':
        heatmap_unnormalized = {} # one heatmap for each metric
        for metric in metrics:
            heatmap_unnormalized[metric] = torch.zeros_like(img)
        heatmap_inclusion_counts = torch.zeros_like(img)
    elif heatmap_type == 'nonaveraged':
        raise NotImplementedError
    
    # anchors for windows
    for h_anchor in tqdm(range(0, img_height-window_size, window_stride), desc = 'Heatmap generating...'):
        for w_anchor in range(0, img_width-window_size, window_stride):
            # load window from full image
            img_window = img[:, :, h_anchor:h_anchor+window_size, w_anchor:w_anchor+window_size]
            # normalize window for inpainting
            img_window = normalize_img(img_window)
            # do inpaintings
            inpaintings = inpainter(img_window, mask, M_inpaint)
            
            # select only inpaintings of the one image
            inpaintings = inpaintings[0]
            
            # compare inpaintings to ground truth with metric of choice
            # get completions only (no surroundings)
            #completion_ground_truth = torch.zeros((1, 1, mask_size, mask_size)).cuda()
            completion_ground_truth = img_window[:, :, (window_size-mask_size)//2:(window_size+mask_size)//2, (window_size-mask_size)//2:(window_size+mask_size)//2]
            completions = inpaintings[:, :, (window_size-mask_size)//2:(window_size+mask_size)//2, (window_size-mask_size)//2:(window_size+mask_size)//2]
            completion_ground_truth_stacked = torch.cat(M_inpaint * [completion_ground_truth])
            
            # compute anomaly score
            for metric in metrics:
                if metric == 'MCD_image':
                    residuals = completion_ground_truth_stacked - completions
                    if normalize_residuals:
                        normalized_residuals = torch.div(residuals, torch.norm(residuals, p=2).item())
                        if torch.isnan(normalized_residuals).any():
                            print('nan alert: not normalizing residuals.')
                            normalized_residuals = residuals

                        residuals = normalized_residuals

                    L2 = torch.norm(residuals, p=2, dim=(1,2,3)) # L2 distances from completions to g.t.
                    score = torch.min(L2)[0].item() # minimum L2 distance

                elif metric == 'MCD_feature':
                    # note: pretrained feature extractor expects input dimensionality of 128*128, so resize if not
                    completions = F.interpolate(completions, size=128)
                    completion_ground_truth_stacked = F.interpolate(completion_ground_truth_stacked, size=128)

                    # extract features from completion images
                    completion_ground_truth_features_stacked = feature_extractor(completion_ground_truth_stacked)
                    completion_ground_truth_features_stacked = torch.flatten(completion_ground_truth_features_stacked, start_dim=1)
                    completion_features = feature_extractor(completions)
                    completion_features = torch.flatten(completion_features, start_dim=1)

                    residuals = completion_features - completion_ground_truth_features_stacked
                    if normalize_residuals:
                        normalized_residuals = torch.div(residuals, torch.norm(residuals, p=2).item())
                        if torch.isnan(normalized_residuals).any():
                            print('nan alert: not normalizing residuals.')
                            normalized_residuals = residuals

                        residuals = normalized_residuals

                    L2 = torch.norm(residuals, p=2, dim=(1)) # L2 distances from completions to g.t.
                    score = torch.min(L2)[0].item() # minimum L2 distance


                # save score to heatmap
                if heatmap_type == 'averaged':
                    heatmap_unnormalized[metric][:, :, h_anchor:h_anchor+window_size, w_anchor:w_anchor+window_size] += score
                    heatmap_inclusion_counts[:, :, h_anchor:h_anchor+window_size, w_anchor:w_anchor+window_size] += 1
                elif heatmap_type == 'nonaveraged':
                    raise NotImplementedError
                
    heatmaps = {}
    for metric in metrics:         
        heatmap = torch.div(heatmap_unnormalized[metric], heatmap_inclusion_counts)
        heatmap = torch.squeeze(heatmap)
    
        # post-process
        heatmap = torch.nan_to_num(heatmap) # fix nans (non-classified pixels)
        heatmap = heatmap - torch.min(heatmap).item() # zero scores
        
        heatmaps[metric] = heatmap
    return heatmaps

# heatmapping code: fixed surroundings
def generate_anomaly_heatmap_nowindow(img, mask_size, mask_stride, M_inpaint, metric, heatmap_type='averaged', normalize_residuals=False):
    ''' 
    generate anomaly heatmap for torch img tensor
    # inputs:   image, 
                mask_size (assuming square mask),
                mask_stride (pixels),
                M_inpaint (int),
                metric (MSE, MCD),
                heatmap_type (averaged or nonaveraged)
                
    # return: heatmap (H x W)
    '''
    img_height = img.shape[2]
    img_width = img.shape[3]
    mask_zero = torch.zeros((1, 1, img_height, img_width))
    mask_zero = mask_zero.cuda()
    
    if heatmap_type == 'averaged':
        heatmap_unnormalized = torch.zeros_like(mask_zero)
        heatmap_inclusion_counts = torch.zeros_like(mask_zero)
    elif heatmap_type == 'nonaveraged':
        raise NotImplementedError
    
    for h_anchor in tqdm(range(0, img_height-mask_size, mask_stride), desc = 'Heatmap generating...'):
        for w_anchor in range(0, img_width-mask_size, mask_stride):
            # create sliding mask
            mask = mask_zero.clone()
            mask[:, :, h_anchor:h_anchor+mask_size, w_anchor:w_anchor+mask_size] = 1.
            
            # do inpaintings
            inpaintings = inpainter(img, mask, M_inpaint)
            
            # compare inpaintings to ground truth with metric of choice
            # get completions only (no surroundings)
            #completion_ground_truth = torch.zeros((1, 1, mask_size, mask_size)).cuda()
            completion_ground_truth = img[:, :, h_anchor:h_anchor+mask_size, w_anchor:w_anchor+mask_size]
            completions = inpaintings[:, :, h_anchor:h_anchor+mask_size, w_anchor:w_anchor+mask_size]
            completion_ground_truth_stacked = torch.cat(M_inpaint * [completion_ground_truth])
            
            # compute anomaly score
            if metric == 'MCD_image':
                residuals = completion_ground_truth_stacked - completions
                if normalize_residuals:
                    normalized_residuals = torch.div(residuals, torch.norm(residuals, p=2).item())
                    if torch.isnan(normalized_residuals).any():
                        print('nan alert: not normalizing residuals.')
                        #print(torch.mean(residuals).item(), torch.std(residuals).item())
                        normalized_residuals = residuals
                             
                    residuals = normalized_residuals
                    
                L2 = torch.norm(residuals, p=2, dim=(1,2,3)) # L2 distances from completions to g.t.
                score = torch.min(L2).item() # minimum L2 distance
                
            elif metric == 'MCD_feature':
                # note: pretrained feature extractor expects input dimensionality of 128*128, so resize if not
                completions = F.interpolate(completions, size=128)
                completion_ground_truth_stacked = F.interpolate(completion_ground_truth_stacked, size=128)
                
                # extract features from completion images
                completion_ground_truth_features_stacked = feature_extractor(completion_ground_truth_stacked)
                completion_ground_truth_features_stacked = torch.flatten(completion_ground_truth_features_stacked, start_dim=1)
                completion_features = feature_extractor(completions)
                completion_features = torch.flatten(completion_features, start_dim=1)
                
                residuals = completion_features - completion_ground_truth_features_stacked
                if normalize_residuals:
                    normalized_residuals = torch.div(residuals, torch.norm(residuals, p=2).item())
                    if torch.isnan(normalized_residuals).any():
                        print('nan alert: not normalizing residuals.')
                        normalized_residuals = residuals

                    residuals = normalized_residuals

                L2 = torch.norm(residuals, p=2, dim=(1)) # L2 distances from completions to g.t.
                score = torch.min(L2).item() # minimum L2 distance
            
            
            # save score to heatmap
            if heatmap_type == 'averaged':
                heatmap_unnormalized[:, :, h_anchor:h_anchor+mask_size, w_anchor:w_anchor+mask_size] += score
                heatmap_inclusion_counts[:, :, h_anchor:h_anchor+mask_size, w_anchor:w_anchor+mask_size] += 1
            elif heatmap_type == 'nonaveraged':
                raise NotImplementedError
                
                
    if heatmap_type == 'averaged':
        heatmap = torch.div(heatmap_unnormalized, heatmap_inclusion_counts)
        heatmap = torch.squeeze(heatmap)
    elif heatmap_type == 'nonaveraged':
        raise NotImplementedError
    
    # post-process
    heatmap = torch.nan_to_num(heatmap) # fix nans (non-classified pixels)
    heatmap = heatmap - torch.min(heatmap).item() # zero scores
    
    return heatmap