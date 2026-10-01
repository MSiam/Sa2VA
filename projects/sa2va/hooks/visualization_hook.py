# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np
import os
import os.path as osp
import warnings
from typing import Optional, Sequence
from PIL import Image
import cv2

import mmcv
from mmengine.fileio import get
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.visualization import Visualizer

from mmengine.registry import HOOKS

def denormalize(img):
    mean = [123.675, 116.28, 103.53]
    scale = [58.395, 57.12, 57.375]

    img = img * torch.tensor(scale) + torch.tensor(mean)
    img = img.cpu().numpy()
    img = np.asarray(img[:,:,:,::-1]*255, np.uint8)
    return img

@HOOKS.register_module()
class SegVisualizationHook(Hook):
    """Segmentation Visualization Hook. Used to visualize validation and
    testing process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(self,
                 draw: bool = False,
                 interval: int = 50,
                 val_interval: int = 50,
                 show: bool = False,
                 wait_time: float = 0.,
                 backend_args: Optional[dict] = None):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.interval = interval
        self.val_interval = val_interval
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.backend_args = backend_args.copy() if backend_args else None
        self.draw = draw
        if not self.draw:
            warnings.warn('The draw is False, it means that the '
                          'hook for visualization will not take '
                          'effect. The results will NOT be '
                          'visualized or stored.')
        self._test_index = 0

    def _show_mask_pred(self, images, masks):
        color = (255, 0, 0)

        final_images = []
        for image, mask in zip(images, masks):
            _mask_image = np.zeros((masks.shape[1], masks.shape[2], 3), dtype=np.uint8)

            _mask_image[:, :, 0] = _mask_image[:, :, 0] + mask.astype(np.uint8) * color[0]
            _mask_image[:, :, 1] = _mask_image[:, :, 1] + mask.astype(np.uint8) * color[1]
            _mask_image[:, :, 2] = _mask_image[:, :, 2] + mask.astype(np.uint8) * color[2]

            overlay_img = image * 0.5 + _mask_image * 0.5
            overlay_img = overlay_img.astype(np.uint8)
            image[mask==1] = overlay_img[mask==1]
            final_images.append(np.array(Image.fromarray(image), np.uint8))

        return final_images

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: dict = None,
                         outputs = None) -> None:
        """Run after every ``self.interval`` train iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]]): A batch of data samples
                that contain annotations and predictions.
        """
        if self.draw is False:
            return

        # There is no guarantee that the same batch of images
        # is visualized for each evaluation.
        total_curr_iter = runner.iter + batch_idx

        # Visualize only the first data

        if total_curr_iter % self.interval == 0:
            log_vars = runner.model.all_logging_vars
            imgs = log_vars['imgs']
            masks = log_vars['masks']
            preds = log_vars['preds']
            exprs = log_vars['expr']
            imgs_llm =  log_vars['imgs_llm'].permute(0,2,3,1)
            imgs_llm = denormalize(imgs_llm)

            # create overlay
            img_grid_arr = []
            for eid, expr in enumerate(exprs):
                try:
                    preds_e = torch.sigmoid(preds[:, eid])
                    preds_e[preds_e < 0.5] = 0
                    preds_e[preds_e >= 0.5] = 1
                    img_masks = self._show_mask_pred(imgs.permute(0,2,3,1).numpy().copy(), masks[:, eid].numpy())
                    img_preds = self._show_mask_pred(imgs.permute(0,2,3,1).numpy().copy(), preds_e.numpy())
                except:
                    continue

                img_grid_exp_arr = []
                for fid, (img_mask, img_pred, img_llm) in enumerate(zip(img_masks, img_preds, imgs_llm)):
                    if fid==0:
                        img_mask = cv2.putText(img_mask, expr.strip().split('assistant')[0], (10,10),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0,0,255), 1, cv2.LINE_AA)

                    img_grid_exp_arr.append(np.concatenate([img_mask, img_pred], axis=1))

                img_grid_exp_arr = np.concatenate(img_grid_exp_arr, axis=0)

                self._visualizer.add_image(
                    'Train Video Expression %d'%eid,
                    image=img_grid_exp_arr,
                    step=total_curr_iter)

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs) -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]]): A batch of data samples
                that contain annotations and predictions.
        """
        if self.draw is False:
            return

        # There is no guarantee that the same batch of images
        # is visualized for each evaluation.
        total_curr_iter = runner.iter + batch_idx

        # Visualize only the first data
        if total_curr_iter % self.val_interval == 0:
            log_vars = runner.model.all_logging_vars
            imgs = log_vars['imgs']
            masks = log_vars['masks']
            preds = log_vars['preds']
            exprs = log_vars['expr']
            imgs_llm =  log_vars['imgs_llm'].permute(0,2,3,1)
            imgs_llm = denormalize(imgs_llm)

            # create overlay
            img_grid_arr = []
            for eid, expr in enumerate(exprs):
                try:
                    preds_e = torch.sigmoid(preds[:, eid])
                    preds_e[preds_e < 0.5] = 0
                    preds_e[preds_e >= 0.5] = 1
                    img_masks = self._show_mask_pred(imgs.permute(0,2,3,1).numpy().copy(), masks[:, eid].numpy())
                    img_preds = self._show_mask_pred(imgs.permute(0,2,3,1).numpy().copy(), preds_e.numpy())
                except:
                    continue

                img_grid_exp_arr = []
                for fid, (img_mask, img_pred, img_llm) in enumerate(zip(img_masks, img_preds, imgs_llm)):
                    if fid==0:
                        img_mask = cv2.putText(img_mask, expr.strip().split('assistant')[0], (10,10),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0,0,255), 1, cv2.LINE_AA)

                    img_grid_exp_arr.append(np.concatenate([img_mask, img_pred], axis=1))

                img_grid_exp_arr = np.concatenate(img_grid_exp_arr, axis=0)

                self._visualizer.add_image(
                    'Val Video Expression %d'%eid,
                    image=img_grid_exp_arr,
                    step=total_curr_iter)


    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs) -> None:
        """Run after every testing iterations.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]): A batch of data samples
                that contain annotations and predictions.
        """
        if self.draw is False:
            return

        for data_sample in outputs:
            self._test_index += 1

            img_path = data_sample.img_path
            window_name = f'test_{osp.basename(img_path)}'

            img_path = data_sample.img_path
            img_bytes = get(img_path, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, channel_order='rgb')

            self._visualizer.add_datasample(
                window_name,
                img,
                data_sample=data_sample,
                show=self.show,
                wait_time=self.wait_time,
                step=self._test_index)
