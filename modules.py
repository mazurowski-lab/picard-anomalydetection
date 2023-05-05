##############################
# utilities for loading neural net models and modules.
##############################
import os
import shutil
import torch
from utils import show_images, count_parameters
import torchvision.utils as vutils
import torchvision.transforms as transforms
import sys
from subprocess import call
from PIL import Image

# model loaders
def load_multi_inpainter(model_type, checkpoints, hyperparams, device_ids, dropoutmodel_config=None, HFPIC_tmp_dir='/workspace/HFPIC_tmp'):
    # load multi-inpainter model (dropout or HFPIC), ready for inference
    # returns a blackbox function that returns $M$ inpainting images: image, mask, M_inpaint -> inpaintings
    # image is (1 x C x H x W) (floats)
    # mask is (1 x 1 x H x W) (binary)
    # (N > 1 not yet implemented)
    use_cuda = False
    if device_ids:
        use_cuda = True
        
    if model_type == 'dropout':
        # import from dropout method (slightly different model if on DBT data)
        # directory and a few commands will be different for DBT data
        # if dataset_name == 'DBT':
        #     raise NotImplementedError 
        # else: # any other dataset (e.g. MVTec benchmark) 
        sys.path.append("inpainter")
        from model.networks import Generator
        from inpainterutils.tools import get_config, get_model_list
        from inpainterutils.dropout import customize_dropout, apply_dropout_on
        # switch back to main dir

        # load config
        config = get_config(dropoutmodel_config)

        # hyperparam set up
        droprate = hyperparams['p_dropout']

        config['netG']['droprate'] = droprate
        if droprate:
            prev_droprate = config['netG']['droprate']

        # send cuda info to config
        config['gpu_ids'] = device_ids

        # load inpainter
        netG = Generator(config['netG'], use_cuda, device_ids)
        netG = netG.to('cuda')
        last_model_name_gen = get_model_list(checkpoints['gen'], "gen", iteration=checkpoints['iter'])
        gen_statedict = torch.load(last_model_name_gen)
        netG.load_state_dict(gen_statedict)
        netG = torch.nn.DataParallel(netG)
        netG.eval()
        
        # print('num params = {}'.format(count_parameters(netG)))

        # apply dropout to inpainter
        if config['netG']['droprate']:
            if config['netG']['dropout_which'] == 'CUSTOM':
                customize_dropout(netG, config)
                # only turn certain dropouts on (to train)
            else:
                # turn on non-custom setting of dropout
                netG.apply(apply_dropout_on)


        def multi_inpainter(image, mask, M_inpaint):
            assert len(image.shape) == 4, 'image shape = {} is does not have length 4.'.format(image.shape)
            assert image.shape[0] == mask.shape[0]
                
            N_inputs = image.shape[0]


            # net can only see non-masked region
            image = image * (1. - mask)


            # make inpaintings (in parallel)
            # duplicate single image along batch dim, M_inpaint times
            # i.e. for M=3, want it to be of the form x1 x1 x1 x2 x2 x2...
            image = torch.unsqueeze(image, dim=1)
            image = torch.cat(M_inpaint*[image], dim=1)
            image = image.view(M_inpaint * N_inputs, *image.shape[2:])
            mask = torch.cat(M_inpaint*[mask]) #stack of N_inputs * M_inpaint
            assert image.shape[0] == M_inpaint * N_inputs

            #show_images(image.cpu())
            
            _, image_reconstruction, _ = netG(image, mask)
            image_inpainted = image_reconstruction * mask + image * (1. - mask)
            inpaintings = image_inpainted # of dim (N_input*M_inpaint) x C x W x W
            
            # resize to be convenient
            inpaintings = torch.reshape(inpaintings, (N_inputs, M_inpaint, inpaintings.shape[1], inpaintings.shape[2], inpaintings.shape[3]))
            return inpaintings # size N_input x M_inpaint x ...
        
    elif model_type == 'HFPIC':
        main_dir = os.getcwd()
        tmp_dir = os.path.join(main_dir, HFPIC_tmp_dir)
        # path to store priors
        prior_url = os.path.join(tmp_dir, 'AP')
        if os.path.exists(prior_url):
            shutil.rmtree(prior_url)
        os.mkdir(prior_url) # create fresh, empty dir
        

        def multi_inpainter(image, mask, M_inpaint):
            assert len(image.shape) == 4, 'image shape = {} is does not have length 4.'.format(image.shape)
            assert image.shape[0] == mask.shape[0]
            
            test_batch_size = M_inpaint
            
            # for now, just run commands instead of using direct python objects since codebase is so complex
            # save imgs in tmp dir
            imgs_url = os.path.join(tmp_dir, 'input_imgs')
            if os.path.exists(imgs_url):
                shutil.rmtree(imgs_url)
            os.mkdir(imgs_url) # create fresh, empty dir
            
            n_input = image.size(0)
            for i in range(n_input):
                save_path = os.path.join(imgs_url, '{}.png'.format(i))
                vutils.save_image(image[i, :, :, :], save_path)
                
            # save mask (duplicated for each image) in tmp dir
            masks_url = os.path.join(tmp_dir, 'input_masks')
            if os.path.exists(masks_url):
                shutil.rmtree(masks_url)
            os.mkdir(masks_url) # create fresh, empty dir
            
            for i in range(n_input):
                # save duplicate mask
                save_path = os.path.join(masks_url, 'mask{}.png'.format(i))
                vutils.save_image(mask[0, :, :, :], save_path)
            
            
            # save images in temporary dir for retrieval (saving and loading operations take far less time than inference anyways)
            
            # STAGE 1: command for using trained transformer to generate priors
            visible_devices = ','.join(str(i) for i in device_ids)
            sys.path.append("src/HFPIC/ICT/Transformer") 
            stage_1_command = "CUDA_VISIBLE_DEVICES=" + visible_devices + " python3 inference.py --ckpt_path " + checkpoints['transformer'] + " \
                                    --BERT --image_url " + imgs_url + " \
                                    --mask_url " + masks_url + " \
                                    --n_layer 35 --n_embd 1024 --n_head 8 --top_k 40 --GELU_2 \
                                    --save_url " + prior_url + " \
                                    --image_size 32 --n_samples " + str(test_batch_size)
            
            run_cmd_HFPIC(stage_1_command)
            print("Finish the Stage 1 - Appearance Priors Reconstruction using Transformer")

            # STAGE 2: use guided upsampler to generate inpaintings with priors
            
            # save output inpaintings (each as a separate img) in tmp dir
            out_url = os.path.join(tmp_dir, 'inpaintings')
            if os.path.exists(out_url):
                shutil.rmtree(out_url)
            os.mkdir(out_url) # create fresh, empty dir
            
            sys.path.append("../Guided_Upsample")
            stage_2_command = "CUDA_VISIBLE_DEVICES=" + visible_devices + " python3 test.py --input " + imgs_url + " \
                                        --mask " + masks_url + " \
                                        --prior " + prior_url + " \
                                        --output " + out_url + " \
                                        --checkpoints ../ckpts_ICT/Upsample/DBT/allmasks_normalval \
                                        --test_batch_size " + str(test_batch_size) + " --model 2 --Generator 4 --condition_num " + str(test_batch_size)# + " --same_face"

            run_cmd_HFPIC(stage_2_command)
            
            print('inpaintings saved to {}.'.format(out_url))
            
            # load saved images to tensors (if it ain't broke don't fix it???)
            all_imgs_iptings = []
            for img_idx in range(n_input):
                img_iptings = []
                for inpainting_idx in range(M_inpaint):
                    ipting_path = os.path.join(out_url, '{}_{}.png'.format(img_idx, inpainting_idx))
                    ipting = Image.open(ipting_path).convert("RGB")
                    ipting = transforms.ToTensor()(ipting) # C x H x W
                    ipting = torch.unsqueeze(ipting, dim=0) # 1 x C x H x W
                    img_iptings.append(ipting)
                img_iptings = torch.cat(img_iptings) # M x C x H x W
                img_iptings = torch.unsqueeze(img_iptings, dim=0) # 1 x M x C x H x W
                all_imgs_iptings.append(img_iptings)
                
            all_imgs_iptings = torch.cat(all_imgs_iptings) # N x M x C x H x W 
            
            return all_imgs_iptings # size N_input x M_inpaint x ...
    
    else:
        raise NotImplementedError
    return multi_inpainter 
    
def load_inpainting_feature_extractor(model_type, checkpoints, hyperparams, device_ids, dropoutmodel_config=None, return_critic_score=False):
    # returns a blackbox function that returns feature representation of completion(s): completion(s) -> feature(s)
    # completion(s) is (N x C x H_completion x W_completion) (floats)
    use_cuda = False
    if device_ids:
        use_cuda = True
        
    if model_type == 'dropout' or model_type == 'HFPIC':
        if model_type == 'HFPIC':
            print('feature extractor not implemented for HFPIC (only use image MSE metric for now).\n using dropout inpainter critic instead')
        
        # import from dropout method (slightly different model if on DBT data)
        # directory and a few commands will be different for DBT data
        sys.path.append("inpainter")
        from model.networks import LocalDis
        from inpainterutils.tools import get_config, get_model_list
        # switch back to main dir

        # load config
        config = get_config(dropoutmodel_config)

        # send cuda info to config
        config['gpu_ids'] = device_ids

        # load critic/feature extractor
        netD_local = LocalDis(config['netD'], use_cuda, device_ids, save_featuremap=True)
        netD_local = netD_local.to('cuda')
        last_model_name_dis = get_model_list(checkpoints['dis'], "dis", iteration=checkpoints['iter'])
        dis_statedict = torch.load(last_model_name_dis)['localD']
        netD_local.load_state_dict(dis_statedict)
        netD_local = torch.nn.DataParallel(netD_local)
        netD_local.eval()

        def extractor(completion):
            # completion is (N x C x H_completion x W_completion) (floats)o
            
            # convert to correct n_channels if needed
            if completion.shape[1] == 3:
                completion = transforms.functional.rgb_to_grayscale(completion)
            
            #features, score = netD_local(completion) # second output is the critic score output
            score = netD_local(completion) # second output is the critic score output
            features = netD_local.module.saved_featuremap
            if return_critic_score:
                return features, score
            else:
                return features
    elif model_type == 'HFPIC':
        print('feature extractor not implemented for HFPIC (only use image MSE metric for now).\n using dropout inpainter critic instead')
        
        return None
    else:
        raise NotImplementedError
        
    return extractor

def run_cmd_HFPIC(command):
    try:
        call(command, shell=True)
    except KeyboardInterrupt:
        print("Process interrupted")
        sys.exit(1)