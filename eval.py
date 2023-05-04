##############################
# tools for evaluating anomaly detection/localization performance.
##############################
import torch
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

fontsize_lrg = 16
fontsize_med = 12
fontsize_sm = 10

# results visualization tools
def plot_score_dists(heatmaps, segmentation, ignore_mask=None, save=False):
    segmentation = segmentation == 1.
    for metric, heatmap in heatmaps.items():
        if ignore_mask is not None:
            gt_cancer_scores = torch.masked_select(heatmap, torch.logical_and(segmentation, ~ignore_mask)).cpu().numpy()
            gt_normal_scores = torch.masked_select(heatmap, torch.logical_and(~segmentation, ~ignore_mask)).cpu().numpy()
        else:
            gt_cancer_scores = torch.masked_select(heatmap, segmentation).cpu().numpy()
            gt_normal_scores = torch.masked_select(heatmap, ~segmentation).cpu().numpy()

        plt.figure(figsize=(4, 3))
        n_n, bins_n, patches_n = plt.hist(gt_normal_scores, color='b', label='negative', alpha=0.75, density=True)
        n_c, bins_c, patches_c = plt.hist(gt_cancer_scores, color='r', label='positive', alpha=0.75, density=True)
        plt.ylabel('normalized counts', fontsize=fontsize_med)
        plt.xlabel('anomaly score', fontsize=fontsize_med)
        plt.ylim(0, 0.08)

        plt.axvline(x=np.mean(gt_normal_scores), linestyle='--')
        plt.axvline(x=np.mean(gt_cancer_scores), linestyle='--', color='r')

        plt.title('Heatmap Pixel Score Distribution\n(normalized)', fontsize=fontsize_lrg)
        plt.legend(fontsize=fontsize_med)
        
        if save:
            savename = 'visualization/output/disteg.pdf'
            plt.savefig(savename, bbox_inches="tight")
        
        plt.show()

    return
            
def plot_roc_curves(heatmaps, segmentation, ignore_mask=None, save=False):
    for metric, heatmap in heatmaps.items():
        # plot roc curve
        if ignore_mask is not None:
            fpr, tpr, thresholds = metrics.roc_curve(
                torch.masked_select(segmentation, ~ignore_mask).cpu().numpy().astype(int).reshape(-1), 
                torch.masked_select(heatmap, ~ignore_mask).cpu().numpy().reshape(-1)
                                                    )
        else:
            fpr, tpr, thresholds = metrics.roc_curve(
                segmentation.cpu().numpy().astype(int).reshape(-1), 
                heatmap.cpu().numpy().reshape(-1)
            )
        auc = metrics.auc(fpr, tpr)
        plt.figure(figsize=(4, 3))
        # plt.title('pixel ROC for heatmap\n with {} metric'.format(metric))
        plt.title('Heatmap Pixel Score ROC', fontsize=fontsize_lrg)
        plt.plot(fpr, tpr, label='AUC = {}'.format(round(auc, 3)))
        plt.plot(tpr, tpr, label='random guessing')
        plt.xlabel('false positive rate', fontsize=fontsize_med)
        plt.ylabel('true positive rate', fontsize=fontsize_med)
        plt.legend(fontsize=fontsize_med)
        
        if save:
            savename = 'visualization/output/ROCeg.pdf'
            plt.savefig(savename, bbox_inches="tight")
        
        plt.show()
    return

            
# heatmap evaluation tools
def score_heatmap(score_type, heatmap, bboxes, ignore_mask=None, threshold_frac=None):
    segmentation = torch.zeros_like(heatmap)
    for bbox in bboxes:
        t, l, h, w = bbox

        segmentation[t:t+h, l:l+w] = 1.
        
    segmentation_original = torch.clone(segmentation)
        
    if ignore_mask is not None:
        segmentation = torch.masked_select(segmentation, ~ignore_mask)
        torch.masked_select(heatmap, ~ignore_mask)
    
    if score_type == 'pixel_AUC': 
        score = metrics.roc_auc_score(
                segmentation.cpu().numpy().astype(int).reshape(-1), 
                heatmap.cpu().numpy().reshape(-1)
        )
        
    elif score_type == 'AP':
        score = metrics.average_precision_score(
                segmentation.cpu().numpy().astype(int).reshape(-1), 
                heatmap.cpu().numpy().reshape(-1)
        )
    
    else:
        raise NotImplementedError
    
    return segmentation_original.cpu(), score