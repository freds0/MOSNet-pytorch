from scipy.stats.stats import _first
import torch
from utils import read_list
import os
import h5py
import numpy as np
import random
import torch.utils.data
import torchaudio
import torchaudio.transforms as T

MAX_WAV_VALUE = 32768.0


def load_wav_to_torch(full_path):
    signal, sr = torchaudio.load(full_path)
    return signal, sr


class getdataset(torch.utils.data.Dataset):
    def __init__(self, config, seed, mode):

        self.config = config
        mos_list = read_list(os.path.join(config["data_dir"], 'mos_list.txt'))
        random.seed(seed)
        random.shuffle(mos_list)
        self.max_wav_value = MAX_WAV_VALUE
        self.spec_fn = T.Spectrogram(
            n_fft=512,
            win_length=512,
            hop_length=256,
            center=True,
            pad_mode="reflect",
            power=2.0,
        )

        self.max_timestep = self.getmax_timestep(config, seed)
        if mode == "train":
            self.filelist = mos_list[0:-(config["num_test"] + config["num_valid"])]
        elif mode == "valid":
            self.filelist = mos_list[-(config["num_test"] + config["num_valid"]):-config["num_test"]]
        elif mode == "test":
            self.filelist = mos_list[-config["num_test"]:]

    def get_magnitude_spec(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        mag_spec = self.spec_fn(audio_norm)
        mag_spec = torch.squeeze(mag_spec, 0)
        return mag_spec

    def read(self, file_path):
        mag_spec = self.get_magnitude_spec(file_path)
        time_step = mag_spec.shape[2]
        spec_dim = self.config["fft_size"] // 2 + 1
        mag_spec = np.reshape(mag_spec,(1, time_step, spec_dim))
        return {
            'mag_spec': mag_spec,
        }

    def pad(self, array, reference_shape):

        result = np.zeros(reference_shape)
        result[:array.shape[0], :array.shape[1], :array.shape[2]] = array

        return result

    def getmax_timestep(self, config, seed):
        file_list = read_list(os.path.join(config["data_dir"], 'mos_list.txt'))
        random.seed(seed)
        random.shuffle(file_list)
        filename = [file_list[x].split(',')[0] for x in range(len(file_list))]
        for i in range(len(filename)):
            all_feat = self.read(os.path.join(config["bin_root"], filename[i]))
            mag_spec = all_feat['mag_spec']
            if i == 0:
                feat = mag_spec
                max_timestep = feat.shape[1]
            else:
                if mag_spec.shape[1] > max_timestep:
                    max_timestep = mag_spec.shape[1]
        return max_timestep

    def __getitem__(self, index):
        # Read audio
        filename, mos = self.filelist[index].split(',')
        all_feat = self.read(os.path.join(self.config["bin_root"], filename))
        mag_spec = all_feat['mag_spec']
        ref_shape = [mag_spec.shape[0], self.max_timestep, mag_spec.shape[2]]
        mag_spec = self.pad(mag_spec, ref_shape)
        mos = np.asarray(float(mos)).reshape([1])
        frame_mos = np.array([mos * np.ones([mag_spec.shape[1], 1])])
        return mag_spec, [mos, frame_mos.reshape((1, -1)).transpose(1, 0)]

    def __len__(self):
        return len(self.filelist)
