
import os 
import numpy as np 
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import torch
import seaborn as sns
from torchvision import datasets
from mmdet.datasets import build_dataset,build_dataloader

# from timm.data import create_loader
from mmcv import Config, DictAction
from tqdm import tqdm
import argparse

score_thr = '30'
res_list = {'DFA':f'DFA_det',
            'FGD':f'FGD_det',
            'STU':f'STU_det',
            'TEA':f'TEA_det'}
out_dir = f'combined_{score_thr}/'

def combine(res_list,out_dir):
    os.makedirs(out_dir,exist_ok=True)
    out_list = {}
    for res_key,res_path in res_list.items():
        # for sub_path in ['good','bad']:
        # _sub_path = os.path.join(res_path,sub_path)
        file_list = os.listdir(res_path)
        for _file in file_list:
            prefix = _file[:12]
            if prefix not in out_list:
                out_list[prefix] = []
            out_list[prefix].append(os.path.join(res_path,_file))

    for key, path_list in out_list.items():
        img_np = None 
        for i, img_path in enumerate(path_list):
            _img = cv2.imread(img_path)
            if img_np is None :
                img_np = np.zeros([_img.shape[0],(_img.shape[1] + 10 )*len(path_list),3], dtype=np.uint8)
            start = (_img.shape[1] + 10 ) * i
            end = start + _img.shape[1]
            img_np[:,start:end,:] = _img.copy()
        cv2.imwrite( os.path.join(out_dir,key+'.jpg'), img_np)

def spect(img):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    return magnitude_spectrum

def visual_freq():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    datasets = build_dataset(cfg.data.test)
    train_dataloader_default_args = dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        # `num_gpus` will be ignored if distributed
        num_gpus=4,
        dist=False,
        seed=0,
        runner_type='EpochBasedRunner',
        persistent_workers=False)
    dataloader = build_dataloader(datasets, **train_dataloader_default_args)
    img_list = []
    for i, data in enumerate(tqdm(dataloader)):

        # images = data['adv'].data.cpu().numpy()
        images = data['img'].data[0][0].numpy().transpose(1,2,0).astype(np.uint8)        
        img_list.append(images)
        if len(img_list) == 256:
            img_np = np.array(img_list)
            mean_spect = np.mean([spect(ch) for img in img_np for ch in img], axis=0)
            plt.figure(figsize=(14.5, 12.5))
            sns_plot= sns.heatmap(
                mean_spect[1:, 1:] / mean_spect[1:, 1:].max(),
                vmin=0,
                vmax=1,
                cmap="jet",
                cbar=True,
                xticklabels=False,
                yticklabels=False,
            )
            cbar = sns_plot.collections[0].colorbar
            # here set the labelsize by 20
            cbar.ax.tick_params(labelsize=40)
            plt.title("Natural Images", fontdict={"size":55})
            plt.tight_layout()
            # plt.savefig("figures/natural_imgs.png")
            plt.savefig("figures/crop_imgs.png")
            # plt.savefig("figures/spectrum_img.png")
            break
        
def err_call_back(err):
    print(f'出错啦~ error: {str(err)}')
def split_and_write(file_name):
    
    img = cv2.imread(file_name)
    
    # 将每个通道分开处理
    b, g, r = cv2.split(img)

    # 对每个通道分别进行低通滤波
    def apply_lowpass_filter(img):
        gray = img
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        fshift = np.fft.fftshift(dft)
        # magnitude_spectrum = np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

        rows, cols = gray.shape
        crow, ccol = rows//2, cols//2

        mask = np.expand_dims(np.zeros((rows, cols), np.uint8),-1)
        mask[crow-30:crow+30, ccol-30:ccol+30] = 1
        
        # total_rows, total_cols = gray.shape
        # X, Y = np.ogrid[:total_rows, :total_cols]
        # center_row, center_col = total_rows/4, total_cols/4
        # dist_from_center = (X - center_row)**2 + (Y - center_col)**2
        # radius = (total_rows/2)**2
        # circular_mask = (dist_from_center > radius)
        
        lp_fshift = fshift * mask
        
        lp_ishift = np.fft.ifftshift(lp_fshift)
        # lp_img_back = np.fft.ifft2(lp_ishift)
        lp_img_back = cv2.idft(lp_ishift)
        lp_img_back = cv2.magnitude(lp_img_back[:,:,0],lp_img_back[:,:,1])
        lp_img_back = np.abs(lp_img_back)
        lp_img_back = (lp_img_back / lp_img_back.max()) * 255.
        lp_img_back = np.uint8(lp_img_back.clip(0,255))
        
        hp_fshift = fshift * (1-mask)
        hp_ishift = np.fft.ifftshift(hp_fshift)
        # hp_img_back = np.fft.ifft2(hp_ishift)
        hp_img_back = cv2.idft(hp_ishift)
        hp_img_back = cv2.magnitude(hp_img_back[:,:,0],hp_img_back[:,:,1])
        hp_img_back = np.abs(hp_img_back)
        hp_img_back = (hp_img_back / hp_img_back.max()) * 255.
        hp_img_back = np.uint8(hp_img_back.clip(0,255))
        

        return lp_img_back,hp_img_back

    b_lp,b_hp = apply_lowpass_filter(b)
    g_lp,g_hp = apply_lowpass_filter(g)
    r_lp,r_hp = apply_lowpass_filter(r)

    # 将每个通道还原成彩色图像

    lp_result = cv2.merge((b_lp, g_lp, r_lp))
    hp_result = cv2.merge((b_hp, g_hp, r_hp))
    cv2.imwrite(file_name.replace('coco/val2017','val2017_lp'), lp_result)
    cv2.imwrite(file_name.replace('coco/val2017','val2017_hp'), hp_result)
    

def split_freq():
    from  multiprocessing import Process,Pool
    cfg = Config.fromfile('/home/sysu/工作目录_梁嘉伟/code/aug/mmdetection/configs/retinanet/retinanet_r50_fpn_2x_coco.py')
    cfg.data.test.test_mode = True
    # cfg.data.test.ann_file='data/coco/annotations/instances_train2017.json'
    # cfg.data.test.img_prefix='data/coco/train2017/'
    datasets = build_dataset(cfg.data.test)
    pool = Pool(128)
    for i, data in enumerate(tqdm(datasets.data_infos)):
        # file_name = data['img_metas'][0].data['filename']
        file_name = os.path.join(datasets.img_prefix,data['file_name'])
        pool.apply_async(func=split_and_write, args=(file_name,),error_callback=err_call_back)
    pool.close()
    pool.join()

if __name__ == '__main__':
    # combine(res_list,out_dir)
    split_freq()
# loader = create_test_dataset()

# for input, _ in loader:

#     images = input.cpu().numpy()
#     print(images.shape)
#     mean_spect = np.mean([spect(ch) for img in images for ch in img], axis=0)

#     plt.figure(figsize=(14.5, 12.5))
#     sns_plot= sns.heatmap(
#         mean_spect[1:, 1:] / mean_spect[1:, 1:].max(),
#         vmin=0,
#         vmax=1,
#         cmap="jet",
#         cbar=True,
#         xticklabels=False,
#         yticklabels=False,
#     )
#     cbar = sns_plot.collections[0].colorbar
#     # here set the labelsize by 20
#     cbar.ax.tick_params(labelsize=40)
#     plt.title("Natural Images", fontdict={"size":55})
#     plt.tight_layout()
#     plt.savefig("figures/spectrum_img.pdf")
#     plt.show()
#     exit()