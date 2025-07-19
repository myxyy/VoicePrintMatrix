import librosa
import os
import numpy as np
from torch.utils.data import TensorDataset
import torch
import pathlib


def JVSDataset(segment_length: int=2048, dataset_path: str='resources/jvs_ver1', seed: int=42) -> TensorDataset:
    num_files = 100
    sample_rate = 22050
    waveform_list = []
    label_list = []

    dataset_path = pathlib.Path(dataset_path)
    filepath_list = []
    for i in range(num_files):
        actor_directory = dataset_path / 'jvs{:0>3}'.format(i+1)
        for root, dirs, files in os.walk(actor_directory):
            for file in files:
                if file.endswith('.wav'):
                    #print(f'{root} {dirs} {file}')
                    filepath = f'{root}/{file}'
                    print(filepath)
                    filepath_list.append(filepath)
                    waveform, sample_rate = librosa.load(filepath)
                    waveform_list.append(waveform)
                    label_list.append(np.full(waveform.shape[0], i, dtype=np.int8))

    # permute waveform_list and label_list in the same order
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(waveform_list))
    waveform_list = [waveform_list[i] for i in permutation]
    label_list = [label_list[i] for i in permutation]
    filepath_list = [filepath_list[i] for i in permutation]

    #for filepath in filepath_list:
    #    print(filepath)

    print('concat_start')
    waveform_concat = np.concatenate(tuple(waveform_list))
    label_concat = np.concatenate(tuple(label_list))
    print('concat_end')
    print('waveform_concat.shape', waveform_concat.shape)
    print('label_concat.shape', label_concat.shape)

    length = waveform_concat.shape[0]
    truncated_length = (length // segment_length) * segment_length

    waveform_array = waveform_concat[0:truncated_length].reshape(truncated_length // segment_length, segment_length)
    label_array = label_concat[0:truncated_length].reshape(truncated_length // segment_length, segment_length)

    waveform_tensor = torch.tensor(waveform_array, dtype=torch.float32)
    label_tensor = torch.tensor(label_array, dtype=torch.int8)
    return TensorDataset(waveform_tensor, label_tensor)
