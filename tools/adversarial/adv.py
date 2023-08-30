import os.path as osp
import pickle
import shutil
import tempfile
import time
import os

import cv2
import mmcv
import torch
import torch.distributed as dist
from .util import get_gt_bboxes_scores_and_labels
from .difgsm import DIFGSM
from .m_difgsm import M_DIFGSM
from .tifgsm import  TIFGSM
from .mifgsm import  MIFGSM
from .vmifgsm import VMIFGSM
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results
ta_factory = {
    'difgsm': DIFGSM,
    'm_difgsm': M_DIFGSM,
    'tifgsm': TIFGSM,
    'mifgsm': MIFGSM,
    'vmifgsm': VMIFGSM
}


def single_gpu_adv(model,
                    data_loader,
                   args):
    model.eval()

    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    attack = ta_factory[args.method](model, args)
    for i, data in enumerate(data_loader):

        adv = attack(data)

        batch_size = adv.shape[0]
        if args.show_dir:
            img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor.detach().clone(), **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                out_file = osp.join(args.show_dir, img_meta['ori_filename'])

                mmcv.imwrite( img_show,out_file)


        for _ in range(batch_size):
            prog_bar.update()


def multi_gpu_adv(model, data_loader, args):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    attack = ta_factory[args.method](model, args)
    for i, data in enumerate(data_loader):

        adv = attack(data)

        batch_size = adv.shape[0]
        if args.show_dir:
            img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor.detach().clone(), **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                out_file = osp.join(args.show_dir, img_meta['ori_filename'])

                mmcv.imwrite(img_show, out_file)

        if rank == 0:

            for _ in range(batch_size * world_size):
                prog_bar.update()


