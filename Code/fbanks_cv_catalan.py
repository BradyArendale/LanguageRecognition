import s3fs
import warnings
import pandas as pd
from pathlib2 import Path
import numpy as np
import random
import tempfile
import librosa
import torch
import torchaudio
import json

# Attach S3 bucket
fs = s3fs.S3FileSystem(key="", 
    secret="")
path = 'agzr-capstone/Data/'

dataset = 'Common_Voice'

# Filter librosa import warnings
warnings.simplefilter('ignore', category=UserWarning)

# Get list of files
files = pd.read_csv('common_voice_files.csv')
# Subset files
language = 'Catalan' # Change this to the language of choice
files = [i for i in files['filename'] if Path(i).parts[1] == language]

# Get speaker csvs
csv_path = Path().joinpath(dataset).joinpath(language).joinpath('validated.tsv')
speaker_df = pd.read_csv(csv_path, sep='\t')

def split_waveform(wav, durs):
    '''Splits a waveform evenly into durations of a number of samples in durs,
    chosen randomly.
    
    Returns a list of lists of equal-length waveforms.'''
    # Get durations shorter than waveform
    num_samples = len(wav)
    wav_durs = [i for i in durs if num_samples//i > 0]
    # Pick a random duration
    dur = random.choice(wav_durs)
    # Calculate number of durations in waveform
    num_durs = num_samples // dur
    # Calculate starting point for each split
    start = (num_samples % dur) // 2
    starts = start + np.array(range(num_durs))*dur
    # Get splits
    return [wav[i:(i+dur)].tolist() for i in starts]

sample_rate = 8000
# Get durations of 2, 5, 10, and 30 seconds at sample rate
durs = np.array([2,5,10,30])*sample_rate
# Initialize lists
languages = []
speakers = []
durations = []
waveforms = []
filterbanks = []

for loop_num, filename in enumerate(files):
    # Open file from S3
    with fs.open(path+filename) as f:
        # Write data to temp file and load waveform
        with tempfile.NamedTemporaryFile() as temp:
            temp.write(f.read())
            try:
                wav, sr = librosa.core.load(temp.name, sr=sample_rate)
            except Exception as e:
                print(f'File {filename} failed to load: {e}')
                continue
    # Get language and speaker
    parts = Path(filename).parts
    try:
        speaker = speaker_df.loc[speaker_df.path == parts[-1], 
                                 'client_id'].item()
        # Speaker is 128 digit hex string, trim it down
        speaker = speaker[-15:]
    except:
        speaker = 'Unknown_Speaker'
    speaker = '-'.join([dataset,speaker])
    # Split waveform
    try:
        splits = split_waveform(wav, durs)
    except IndexError:
        # Waveform is too short
        continue
    for split in splits:
        # Add metadata and waveform
        languages.append(language)
        speakers.append(speaker)
        durations.append(len(split)/sr)
        waveforms.append(split)
        # Compute filterbanks
        filterbank = torchaudio.compliance.kaldi.fbank(torch.tensor(split)[None], 
                                                       sample_frequency=sr,
                                                       num_mel_bins=64)
        # Mean normalization
        filterbank = filterbank - torch.mean(filterbank)
        filterbanks.append(filterbank.t().tolist())
    if (loop_num+1) % 1000 == 0:
        print(f'Finished extracting {loop_num+1} filterbanks')
    if (loop_num+1) % 5000 == 0 or loop_num+1 == len(files):
        # Write to JSON file
        fb_dict = {'language':languages,
                'speaker':speakers,
                'duration':durations,
                'waveform':waveforms,
                'filterbank':filterbanks
                }
        with open(''.join(['filterbanks/',dataset,'_',language,'_filterbanks',
                           str(loop_num+1),'.json']), 'w') as f:
            json.dump(fb_dict, f)
            # Empty lists
            del fb_dict
            languages = []
            speakers = []
            durations = []
            waveforms = []
            filterbanks = []
            print('Wrote JSON')