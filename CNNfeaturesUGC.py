"""Extracting Content-Aware Perceptual Features using Pre-Trained ResNet-50"""
# Modify: Xinyi Wang
# Date: 2021/08/31
#

import skvideo
# skvideo.setFFmpegPath('/mnt/storage/software/apps/ffmpeg-4.3/bin/ffmpeg')

import torch
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import Dataset
import skvideo.io
from PIL import Image
import os
import h5py
import numpy as np
import random
from argparse import ArgumentParser


class VideoDataset(Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, videos_dir, video_names, score, width, height, video_format='RGB'):

        super(VideoDataset, self).__init__()
        self.videos_dir = videos_dir
        self.video_names = video_names
        self.score = score
        self.format = video_format
        self.width = width
        self.height = height

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_name = self.video_names[idx] + '.mkv'
        print(video_name)
        video_width = int(self.width[idx])
        video_height = int(self.height[idx])
        assert self.format == 'YUV420' or self.format == 'RGB'
        if self.format == 'YUV420':
            video_data = skvideo.io.vread(os.path.join(self.videos_dir, video_name), video_height, video_width, inputdict={'-pix_fmt':'yuvj420p'})
            # print(video_data)
        else:
            video_data = skvideo.io.vread(os.path.join(self.videos_dir, video_name))
            # print(video_data)
        video_score = self.score[idx]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        video_length = video_data.shape[0]
        video_channel = video_data.shape[3]
        video_height = video_data.shape[1]
        video_width = video_data.shape[2]
        transformed_video = torch.zeros([video_length, video_channel,  video_height, video_width])
        for frame_idx in range(video_length):
            frame = video_data[frame_idx]
            frame = Image.fromarray(frame)
            frame = transform(frame)
            transformed_video[frame_idx] = frame

        sample = {'video': transformed_video,
                  'score': video_score}

        return sample


class ResNet50(torch.nn.Module):
    """Modified ResNet50 for feature extraction"""
    def __init__(self):
        super(ResNet50, self).__init__()
        self.features = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        # features@: 7->res5c
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii == 7:
                features_mean = nn.functional.adaptive_avg_pool2d(x, 1)
                features_std = global_std_pool2d(x)
                return features_mean, features_std


def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)


def get_features(video_data, frame_batch_size=32, device='cuda'):
    """feature extraction"""
    extractor = ResNet50().to(device)
    video_length = video_data.shape[0]
    frame_start = 0
    frame_end = frame_start + frame_batch_size
    output1 = torch.Tensor().to(device)
    output2 = torch.Tensor().to(device)
    extractor.eval()
    with torch.no_grad():
	    while frame_end < video_length:
	        batch = video_data[frame_start:frame_end].to(device)
	        features_mean, features_std = extractor(batch)
	        output1 = torch.cat((output1, features_mean), 0)
	        output2 = torch.cat((output2, features_std), 0)
	        frame_end += frame_batch_size
	        frame_start += frame_batch_size

	    last_batch = video_data[frame_start:video_length].to(device)
	    features_mean, features_std = extractor(last_batch)
	    output1 = torch.cat((output1, features_mean), 0)
	    output2 = torch.cat((output2, features_std), 0)
	    output = torch.cat((output1, output2), 1).squeeze()

    return output


if __name__ == "__main__":
    parser = ArgumentParser(description='"Extracting Content-Aware Perceptual Features using Pre-Trained ResNet-50')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--database', default='YOUTUBE_UGC_TEST', type=str,
                        help='database name (default: KoNViD-1k)')
    parser.add_argument('--frame_batch_size', type=int, default=64,
                        help='frame batch size for feature extraction (default: 64)')

    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    args = parser.parse_args()

    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    if args.database == 'YOUTUBE_UGC_360P':
        videos_dir = '/mnt/storage/home/um20242/scratch/ugc-dataset/360P/'
        features_dir = '/mnt/storage/home/um20242/scratch/VSFA-UGC/CNN_features_360P/'
        datainfo = '/mnt/storage/home/um20242/scratch/VSFA-UGC/data/YOUTUBE_UGC_360P_info.mat'
    if args.database == 'YOUTUBE_UGC_480P':
        videos_dir = '/mnt/storage/home/um20242/scratch/ugc-dataset/480P/'
        features_dir = '/mnt/storage/home/um20242/scratch/VSFA-UGC/CNN_features_480P/'
        datainfo = '/mnt/storage/home/um20242/scratch/VSFA-UGC/data/YOUTUBE_UGC_480P_info.mat'
    if args.database == 'YOUTUBE_UGC_720P':
        videos_dir = '/mnt/storage/home/um20242/scratch/ugc-dataset/720P/'
        features_dir = '/mnt/storage/home/um20242/scratch/VSFA-UGC/CNN_features_720P/'
        datainfo = '/mnt/storage/home/um20242/scratch/VSFA-UGC/data/YOUTUBE_UGC_720P_info.mat'
    if args.database == 'YOUTUBE_UGC_1080P':
        videos_dir = '/mnt/storage/home/um20242/scratch/ugc-dataset/1080P/'
        features_dir = '/mnt/storage/home/um20242/scratch/VSFA-UGC/CNN_features_1080P/'
        datainfo = '/mnt/storage/home/um20242/scratch/VSFA-UGC/data/YOUTUBE_UGC_1080P_info.mat'
    if args.database == 'YOUTUBE_UGC_2160P':
        videos_dir = '/mnt/storage/home/um20242/scratch/ugc-dataset/2160P/'
        features_dir = '/mnt/storage/home/um20242/scratch/VSFA-UGC/CNN_features_2160P/'
        datainfo = '/mnt/storage/home/um20242/scratch/VSFA-UGC/data/YOUTUBE_UGC_2160P_info.mat'

    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")

    Info = h5py.File(datainfo, 'r')
    video_names = [Info[Info['video_names'][0, :][i]][()].tobytes()[::2].decode() for i in range(len(Info['video_names'][0, :]))]
    scores = Info['scores'][0, :]
    video_format = 'RGB'
    # video_format = Info['video_format'][()].tobytes()[::2].decode()
    print('.................')
    print(video_format)
    width = Info['width'][0, :]
    height = Info['height'][0, :]

    video_list = []
    scores_list = []
    width_list = []
    height_list = []

    for j in range(len(video_names)):
        video = videos_dir + video_names[j] + '.mkv'
        if os.path.isfile(video):
            video_list.append(video_names[j])
            scores_list.append(scores[j])
            width_list.append(width[j])
            height_list.append(height[j])
    dataset = VideoDataset(videos_dir, video_list, scores_list, width_list, height_list, video_format)

    for i in range(len(dataset)):
        current_data = dataset[i]
        # print(current_data)
        current_video = current_data['video']
        current_score = current_data['score']
        print('Video {}: length {}'.format(i, current_video.shape[0]))
        features = get_features(current_video, args.frame_batch_size, device)
        np.save(features_dir + str(i) + '_resnet-50_res5c', features.to('cpu').numpy())
        np.save(features_dir + str(i) + '_score', current_score)
