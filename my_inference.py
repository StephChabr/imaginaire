import argparse
import numpy as np
import cv2
import torch
from PIL import Image

from imaginaire.config import Config
from imaginaire.utils.cudnn import init_cudnn
from imaginaire.utils.dataset import get_test_dataloader
from imaginaire.utils.logging import init_logging
from imaginaire.utils.trainer import \
    (get_model_optimizer_and_scheduler, get_trainer, set_random_seed)

import torchvision.transforms as transforms
from imaginaire.utils.misc import to_device

def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config',
                        help='Path to the training config file.')
    parser.add_argument('--checkpoint', default='',
                        help='Checkpoint path.')
    parser.add_argument('--output_dir',
                        help='Location to save the image outputs')
    parser.add_argument('--logdir',
                        help='Dir for saving logs and models.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed.')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument('--num_workers', type=int)
    args = parser.parse_args()
    return args


def initialise():
    args = parse_args()
    args.config = r'C:\Users\schabril\Desktop\project\pix2pix\vid2vid\test_config.yaml'
    args.output_dir = r'E:\Stephane\spade\test4'
    args.single_gpu = True
    args.num_workers = 0
    #args.checkpoint = r'E:\Stephane\spade\test4\epoch_00150_iteration_000150000_checkpoint.pt'
    args.checkpoint = r'E:\Stephane\spade\cluster\epoch_00145_iteration_000085985_checkpoint.pt'
    #args.checkpoint = r'E:\Stephane\spade\results\test_concat\epoch_00400_iteration_000236000_checkpoint.pt'
    
    set_random_seed(args.seed, by_rank=True)
    cfg = Config(args.config)
    if not hasattr(cfg, 'inference_args'):
        cfg.inference_args = None


    # Override the number of data loading workers if necessary
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers

    # Create log directory for storing training results.
    cfg.date_uid, cfg.logdir = init_logging(args.config, args.logdir)

    # Initialize cudnn.
    init_cudnn(cfg.cudnn.deterministic, cfg.cudnn.benchmark)

    # Initialize data loaders and models.
    test_data_loader = get_test_dataloader(cfg)
    net_G, net_D, opt_G, opt_D, sch_G, sch_D = \
        get_model_optimizer_and_scheduler(cfg, seed=args.seed)
    trainer = get_trainer(cfg, net_G, net_D,
                          opt_G, opt_D,
                          sch_G, sch_D,
                          None, test_data_loader)


    transform_list = [transforms.ToTensor()]
    transform_list.append(
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)))
    transform_im = transforms.Compose(transform_list)
    
    transform_seg = transforms.ToTensor()

    # Load checkpoint.
    trainer.load_checkpoint(cfg, args.checkpoint)

    # Do inference.
    trainer.current_epoch = -1
    trainer.current_iteration = -1
    
    if trainer.cfg.trainer.model_average:
            net_G = trainer.net_G.module.averaged_model
    else:
            net_G = trainer.net_G.module
    net_G.eval()
    
    return net_G, transform_im, transform_seg, cfg
    
    


if __name__ == "__main__":
    net_G, transform_im, transform_seg, cfg = initialise()
    
    img = cv2.imread(r'D:\schabril\Documents\results\test_sequences\segmaps\seq1\render.00003.tif',-1)
    img = img[:, :512, ::-1]
    data = dict()
    data['label'] = img.astype(np.float32)/65535.
    data['label'] = transform_seg((data['label']))
    data['label'] = torch.unsqueeze(data['label'],0)
    
    '''img = cv2.imread(r'E:\Stephane\spade\data\val\images\render.09512.tif',-1)
    img = img[:, :512, ::-1]'''
    img = img = np.load('D:\schabril\Documents\results\test_sequences\images\seq1\render.00003.npy')
    data['images'] = img[:,:512,:]
    #data['images'] = transform_im((data['images']).copy())
    data['images'] = transform_seg((data['images']).copy())
    data['images'] = torch.unsqueeze(data['images'],0)
    
    data['key'] = {'seg_maps':['']}
    
    data = to_device(data, 'cuda')
    with torch.no_grad():
        output_images, file_names = \
            net_G.inference(data, **vars(cfg.inference_args))
        output_images, file_names = \
            net_G.inference(data, **vars(cfg.inference_args))
            
    image = (output_images[0].clamp_(-1, 1) + 1) * 0.5
    image = image.detach().cpu().squeeze().numpy()
    image = np.transpose(image, (1, 2, 0)) * 255
    output_img = Image.fromarray(np.uint8(image))
    
    output_img.show()