import random
from random import sample
import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, resnet18
import mvtec as mvtec
from mytime import get_current_time
import boto3
import shutil


class Config(object):
    def __init__(self):
        self.arch = "wide_resnet50_2"
        self.data_path = "../dataset/mvtec_anomaly_detection/bottle"

class DetectionSystem(object):
    def __init__(self):
        self.s3_client = boto3.client("s3")
        # extract train set features
        train_feature_filepath = 'train.pkl'
        print('load train set feature from: %s' % train_feature_filepath)
        with open(train_feature_filepath, 'rb') as f:
            self.train_outputs = pickle.load(f)
        
    def predict(self,data_path,upload_bucket,upload_path,threshold=0.673):
        args = Config()
        save_path = data_path+"save"
 
        use_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        # load model
        if args.arch == 'resnet18':
            model = resnet18(pretrained=True, progress=True)
            t_d = 448
            d = 100
        elif args.arch == 'wide_resnet50_2':
            model = wide_resnet50_2(pretrained=True, progress=True)
            t_d = 1792
            d = 550
        model.to(device)
        model.eval()
        random.seed(1024)
        torch.manual_seed(1024)
        if use_cuda:
            torch.cuda.manual_seed_all(1024)

        idx = torch.tensor(sample(range(0, t_d), d))

        # set model's intermediate outputs
        outputs = []

        def hook(module, input, output):
            outputs.append(output)

        model.layer1[-1].register_forward_hook(hook)
        model.layer2[-1].register_forward_hook(hook)
        model.layer3[-1].register_forward_hook(hook)

        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

        test_dataset = mvtec.MVTecDataset(data_path, is_train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True)
        test_imgs = []

        # extract test set features
        for (x, y, mask) in test_dataloader:
            test_imgs.extend(x.cpu().detach().numpy())
            # model prediction
            with torch.no_grad():
                _ = model(x.to(device))
            # get intermediate layer outputs
            for k, v in zip(test_outputs.keys(), outputs):
                test_outputs[k].append(v.cpu().detach())
            # initialize hook outputs
            outputs = []
        for k, v in test_outputs.items():
            test_outputs[k] = torch.cat(v, 0)

        # Embedding concat
        embedding_vectors = test_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = self.embedding_concat(embedding_vectors, test_outputs[layer_name])

        # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, idx)

        # calculate distance matrix
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
        dist_list = []
        for i in range(H * W):
            mean = self.train_outputs[0][:, i]
            conv_inv = np.linalg.inv(self.train_outputs[1][:, :, i])
            dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)

        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

        # upsample
        dist_list = torch.tensor(dist_list)
        score_map = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear',
                                  align_corners=False).squeeze().numpy()

        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)

        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)

        os.makedirs(save_path, exist_ok=True)
        self.plot_fig(test_imgs, scores, threshold, save_path)
        
        self.upload(save_path,upload_bucket,upload_path)
        shutil.rmtree(data_path)
        shutil.rmtree(save_path)
        print(data_path+"推理完毕")
        
        return "{'result':'OK'}"

    def upload(self,save_path,upload_bucket,upload_path):
        if not upload_path.endswith("/"):
            upload_path = upload_path + "/"
        for f in os.listdir(save_path):
            file_name = os.path.join(save_path,f)
            self.s3_client.upload_file(file_name,upload_bucket,upload_path+f)

    def plot_fig(self,test_img, scores, threshold, save_dir):
        num = len(scores)
        vmax = scores.max() * 255.
        vmin = scores.min() * 255.
        for i in range(num):
            img = test_img[i]
            img = self.denormalization(img)
            heat_map = scores[i] * 255
            mask = scores[i]
            mask[mask > threshold] = 1
            mask[mask <= threshold] = 0
            kernel = morphology.disk(4)
            mask = morphology.opening(mask, kernel)
            mask *= 255
            vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
            fig_img, ax_img = plt.subplots(1, 4, figsize=(10, 3))
            fig_img.subplots_adjust(right=0.9)
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            for ax_i in ax_img:
                ax_i.axes.xaxis.set_visible(False)
                ax_i.axes.yaxis.set_visible(False)
            ax_img[0].imshow(img)
            ax_img[0].title.set_text('Image')
            ax = ax_img[1].imshow(heat_map, cmap='jet', norm=norm)
            ax_img[1].imshow(img, cmap='gray', interpolation='none')
            ax_img[1].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
            ax_img[1].title.set_text('Predicted heat map')
            ax_img[2].imshow(mask, cmap='gray')
            ax_img[2].title.set_text('Predicted mask')
            ax_img[3].imshow(vis_img)
            ax_img[3].title.set_text('Segmentation result')
            left = 0.92
            bottom = 0.15
            width = 0.015
            height = 1 - 2 * bottom
            rect = [left, bottom, width, height]
            cbar_ax = fig_img.add_axes(rect)
            cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
            cb.ax.tick_params(labelsize=8)
            font = {
                'family': 'serif',
                'color': 'black',
                'weight': 'normal',
                'size': 8,
            }
            cb.set_label('Anomaly Score', fontdict=font)

            fig_img.savefig(os.path.join(save_dir, '{}'.format(i)), dpi=100)
            plt.close()


    def denormalization(self,x):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)

        return x


    def embedding_concat(self,x, y):
        B, C1, H1, W1 = x.size()
        _, C2, H2, W2 = y.size()
        s = int(H1 / H2)
        x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
        x = x.view(B, C1, -1, H2, W2)
        z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
        for i in range(x.size(2)):
            z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
        z = z.view(B, -1, H2 * W2)
        z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

        return z

