# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import functools
import math

import torch
import torch.nn.functional as F

from imaginaire.evaluation import compute_fid
from imaginaire.losses import (FeatureMatchingLoss, GANLoss, GaussianKLLoss,
                               PerceptualLoss)
from imaginaire.trainers.base import BaseTrainer
from imaginaire.utils.distributed import master_only_print as print
from imaginaire.utils.meters import Meter
from imaginaire.utils.model_average import reset_batch_norm, \
    calibrate_batch_norm_momentum
from imaginaire.utils.misc import split_labels, to_device
from imaginaire.utils.visualization import tensor2label


class Trainer(BaseTrainer):
    r"""Initialize SPADE trainer.

    Args:
        cfg (Config): Global configuration.
        net_G (obj): Generator network.
        net_D (obj): Discriminator network.
        opt_G (obj): Optimizer for the generator network.
        opt_D (obj): Optimizer for the discriminator network.
        sch_G (obj): Scheduler for the generator optimizer.
        sch_D (obj): Scheduler for the discriminator optimizer.
        train_data_loader (obj): Train data loader.
        val_data_loader (obj): Validation data loader.
    """

    def __init__(self,
                 cfg,
                 net_G,
                 net_D,
                 opt_G,
                 opt_D,
                 sch_G,
                 sch_D,
                 train_data_loader,
                 val_data_loader):
        super(Trainer, self).__init__(cfg, net_G, net_D, opt_G,
                                      opt_D, sch_G, sch_D,
                                      train_data_loader, val_data_loader)
        if cfg.data.type == 'imaginaire.datasets.paired_videos':
            self.video_mode = True
        else:
            self.video_mode = False

    def _init_loss(self, cfg):
        r"""Initialize loss terms.

        Args:
            cfg (obj): Global configuration.
        """
        self.criteria['GAN'] = GANLoss(cfg.trainer.gan_mode)
        self.weights['GAN'] = cfg.trainer.loss_weight.gan
        # Setup the perceptual loss. Note that perceptual loss can run in
        # fp16 mode for additional speed. We find that running on fp16 mode
        # leads to improve training speed while maintaining the same accuracy.
        if hasattr(cfg.trainer, 'perceptual_loss'):
            self.criteria['Perceptual'] = \
                PerceptualLoss(
                    cfg=cfg,
                    network=cfg.trainer.perceptual_loss.mode,
                    layers=cfg.trainer.perceptual_loss.layers,
                    weights=cfg.trainer.perceptual_loss.weights)
            self.weights['Perceptual'] = cfg.trainer.loss_weight.perceptual
        # Setup the feature matching loss.
        self.criteria['FeatureMatching'] = FeatureMatchingLoss()
        self.weights['FeatureMatching'] = \
            cfg.trainer.loss_weight.feature_matching
        # Setup the Gaussian KL divergence loss.
        self.criteria['GaussianKL'] = GaussianKLLoss()
        self.weights['GaussianKL'] = cfg.trainer.loss_weight.kl

    def _init_tensorboard(self):
        r"""Initialize the tensorboard. For the SPADE model, we will record
        regular and FID, which is the average FID.
        """
        self.regular_fid_meter = Meter('FID/regular')
        if self.cfg.trainer.model_average:
            self.average_fid_meter = Meter('FID/average')
        self.image_meter = Meter('images')
        self.meters = {}
        names = ['optim/gen_lr', 'optim/dis_lr',
                 'time/iteration', 'time/epoch']
        for name in names:
            self.meters[name] = Meter(name)

    def _start_of_iteration(self, data, current_iteration):
        r"""Model specific custom start of iteration process. We will do two
        things. First, put all the data to GPU. Second, we will resize the
        input so that it becomes multiple of the factor for bug-free
        convolutional operations. This factor is given by the yaml file.
        E.g., base = getattr(self.net_G, 'base', 32)

        Args:
            data (dict): The current batch.
            current_iteration (int): The iteration number of the current batch.
        """
        if len(data['label'].size()) == 5:
            label_image_raw = data['images'][:, 0:-1, :, :, :]
            label_image = label_image_raw.reshape(
                [label_image_raw.size(0),
                 -1,
                 label_image_raw.size(3),
                 label_image_raw.size(4)])
            images = data['images'][:, -1, :, :, :]
            label_label = data['label'].reshape(
                [data['label'].size(0),
                 -1,
                 data['label'].size(3),
                 data['label'].size(4)])
            label = torch.cat([label_label, label_image], 1)
            data['label'] = label
            data['images'] = images
        data = to_device(data, 'cuda')
        data = self._resize_data(data)
        return data

    def gen_forward(self, data):
        r"""Compute the loss for SPADE generator.

        Args:
            data (dict): Training data at the current iteration.
        """
        net_G_output = self.net_G(data)
        net_D_output = self.net_D(data, net_G_output)

        self._time_before_loss()

        output_fake = self._get_outputs(net_D_output, real=False)
        self.gen_losses['GAN'] = \
            self.criteria['GAN'](output_fake, True, dis_update=False)

        self.gen_losses['FeatureMatching'] = self.criteria['FeatureMatching'](
            net_D_output['fake_features'], net_D_output['real_features'])

        if self.net_G_module.use_style_encoder:
            self.gen_losses['GaussianKL'] = \
                self.criteria['GaussianKL'](net_G_output['mu'],
                                            net_G_output['logvar'])
        else:
            self.gen_losses['GaussianKL'] = \
                self.gen_losses['GAN'].new_tensor([0])

        if hasattr(self.cfg.trainer, 'perceptual_loss'):
            self.gen_losses['Perceptual'] = self.criteria['Perceptual'](
                net_G_output['fake_images'], data['images'])

        total_loss = self.gen_losses['GAN'].new_tensor([0])
        for key in self.criteria:
            total_loss += self.gen_losses[key] * self.weights[key]

        self.gen_losses['total'] = total_loss
        return total_loss

    def dis_forward(self, data):
        r"""Compute the loss for SPADE discriminator.

        Args:
            data (dict): Training data at the current iteration.
        """
        with torch.no_grad():
            net_G_output = self.net_G(data)
            net_G_output['fake_images'] = net_G_output['fake_images'].detach()
        net_D_output = self.net_D(data, net_G_output)

        self._time_before_loss()

        output_fake = self._get_outputs(net_D_output, real=False)
        output_real = self._get_outputs(net_D_output, real=True)
        fake_loss = self.criteria['GAN'](output_fake, False, dis_update=True)
        true_loss = self.criteria['GAN'](output_real, True, dis_update=True)
        self.dis_losses['GAN/fake'] = fake_loss
        self.dis_losses['GAN/true'] = true_loss
        self.dis_losses['GAN'] = fake_loss + true_loss
        total_loss = self.dis_losses['GAN'] * self.weights['GAN']
        self.dis_losses['total'] = total_loss
        return total_loss

    def _get_visualizations(self, data):
        r"""Compute visualization image. We will first recalculate the batch
        statistics for the moving average model.

        Args:
            data (dict): The current batch.
        """
        self.recalculate_model_average_batch_norm_statistics(
            self.train_data_loader)
        with torch.no_grad():
            label_lengths = self.train_data_loader.dataset.get_label_lengths()
            labels = split_labels(data['label'], label_lengths)
            # Get visualization of the segmentation mask.
            #segmap = tensor2label(labels['seg_maps'],
            #                      label_lengths['seg_maps'],
            #                      output_normalized_tensor=True)
            segmap = labels['segmaps']
            #segmap = torch.cat([x.unsqueeze(0) for x in segmap], 0)
            net_G_output = self.net_G(data, random_style=True)
            vis_images = [data['images'][:,:3,:,:],
                          segmap,
                          net_G_output['fake_images'][:,:3,:,:]]
            if self.cfg.trainer.model_average:
                net_G_model_average_output = \
                    self.net_G.module.averaged_model(data, random_style=True)
                vis_images.append(net_G_model_average_output['fake_images'][:,:3,:,:])
        return vis_images

    def recalculate_model_average_batch_norm_statistics(self, data_loader):
        r"""Update the statistics in the moving average model.

        Args:
            data_loader (pytorch data loader): Data loader for estimating the
                statistics.
        """
        if not self.cfg.trainer.model_average:
            return
        model_average_iteration = \
            self.cfg.trainer.model_average_batch_norm_estimation_iteration
        if model_average_iteration == 0:
            return
        with torch.no_grad():
            # Accumulate bn stats..
            self.net_G.module.averaged_model.train()
            # Reset running stats.
            self.net_G.module.averaged_model.apply(reset_batch_norm)
            for cal_it, cal_data in enumerate(data_loader):
                if cal_it >= model_average_iteration:
                    print('Done with {} iterations of updating batch norm '
                          'statistics'.format(model_average_iteration))
                    break
                # cal_data = to_device(cal_data, 'cuda')
                cal_data = self._start_of_iteration(cal_data, 0)
                # Averaging over all batches
                self.net_G.module.averaged_model.apply(
                    calibrate_batch_norm_momentum)
                self.net_G.module.averaged_model(cal_data)

    def write_metrics(self):
        r"""If moving average model presents, we have two meters one for
        regular FID and one for average FID. If no moving average model,
        we just report average FID.
        """
        if self.cfg.trainer.model_average:
            regular_fid, average_fid = self._compute_fid()
            self.regular_fid_meter.write(regular_fid)
            self.average_fid_meter.write(average_fid)
            meters = [self.regular_fid_meter, self.average_fid_meter]
        else:
            regular_fid = self._compute_fid()
            self.regular_fid_meter.write(regular_fid)
            meters = [self.regular_fid_meter]
        for meter in meters:
            meter.flush(self.current_iteration)

    def _compute_fid(self):
        r"""We will compute FID for the regular model using the eval mode.
        For the moving average model, we will use the eval mode.
        """
        self.net_G.eval()
        net_G_for_evaluation = \
            functools.partial(self.net_G, random_style=True)
        regular_fid_path = self._get_save_path('regular_fid', 'npy')
        preprocess = \
            functools.partial(self._start_of_iteration, current_iteration=0)

        regular_fid_value = compute_fid(regular_fid_path,
                                        self.val_data_loader,
                                        net_G_for_evaluation,
                                        preprocess=preprocess)
        print('Epoch {:05}, Iteration {:09}, Regular FID {}'.format(
            self.current_epoch, self.current_iteration, regular_fid_value))
        if self.cfg.trainer.model_average:
            avg_net_G_for_evaluation = \
                functools.partial(self.net_G.module.averaged_model,
                                  random_style=True)
            fid_path = self._get_save_path('average_fid', 'npy')
            fid_value = compute_fid(fid_path, self.val_data_loader,
                                    avg_net_G_for_evaluation,
                                    preprocess=preprocess)
            print('Epoch {:05}, Iteration {:09}, FID {}'.format(
                self.current_epoch, self.current_iteration, fid_value))
            self.net_G.float()
            return regular_fid_value, fid_value
        else:
            self.net_G.float()
            return regular_fid_value

    def _resize_data(self, data):
        r"""Resize input label maps and images so that it can be properly
        generated by the generator.

        Args:
            data (dict): Input dictionary contains 'label' and 'image fields.
        """
        base = getattr(self.net_G, 'base', 32)
        sy = math.floor(data['label'].size()[2] * 1.0 // base) * base
        sx = math.floor(data['label'].size()[3] * 1.0 // base) * base
        data['label'] = F.interpolate(
            data['label'], size=[sy, sx], mode='nearest')
        if 'images' in data.keys():
            data['images'] = F.interpolate(
                data['images'], size=[sy, sx], mode='bicubic')
        return data
