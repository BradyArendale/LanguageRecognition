from time import time
from pathlib2 import Path
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
import torch

start_time = time()
path = Path().joinpath('filterbanks')

# Load JSONs and append to dataframe
jsons = list(path.iterdir())
filterbanks = pd.DataFrame()

for i in jsons:
    json_fb = pd.read_json(i)
    filterbanks = filterbanks.append(pd.DataFrame(json_fb), ignore_index=True)
    print("Loop complete, time elapsed:", round((time()-start_time)/60, 2), "minutes")
print("Joined JSONs, time elapsed:", round((time()-start_time)/60, 2), "minutes")

# Calculate durations by language
durations = filterbanks.groupby('language').sum()
# Convert to hours
durations = durations/3600
# Calculate percent of total
durations['percent'] = durations['duration']/sum(durations['duration'])
durations.to_csv('language_durations.csv')

# Get unique speakers
speakers = set(zip(filterbanks['language'], filterbanks['speaker']))
speakers = pd.DataFrame(speakers, columns=['language','speaker'])
# Remove unknown speakers
speakers = speakers[~speakers['speaker'].str.contains('Unknown_Speaker', regex=False)]
# Randomly assign train, validation, and test splits (70-15-15)
speakers['split'] = random.choices(['train','valid','test'], cum_weights=[70,85,100],
                                   k=len(speakers))
# Combine splits into main dataframe
filterbanks = pd.merge(filterbanks, speakers, how='left')
# Randomly assign splits to unknown speakers
filterbanks.loc[pd.isnull(filterbanks['split']), 'split'] = random.choices(['train','valid','test'], 
                                                                           cum_weights=[70,85,100], 
                                                                           k=sum(pd.isnull(filterbanks['split'])))
print("Number of unique speakers:", len(speakers))
print("Assigned splits, time elapsed:", round((time()-start_time)/60, 2), "minutes")

# Check split size
print("\nTotal split numbers:")
print(filterbanks['split'].value_counts())
# Check splits per language
print("\nSplits per language:")
print(filterbanks.groupby('language')['split'].value_counts(normalize=True))
print("Time elapsed:", round((time()-start_time)/60, 2), "minutes")

# Group by split
groups = filterbanks.groupby('split')
tensor_path = Path().joinpath('tensors')

# Create integer encoding for labels
lb = LabelEncoder()
labels = ['Bengali', 'Catalan', 'English', 'French', 'German', 'Kabyle', 'Persian', 'Spanish']
lb.fit(labels)
lb_classes = pd.DataFrame({'language':lb.classes_})
lb_classes.to_csv('language_label_encoding.csv')

for name, group in groups:
    # Save filterbank tensors
    fb_tensor = torch.tensor(group['filterbank'].to_list())
    fb_path = tensor_path.joinpath(''.join(['filterbanks_',name,'.pt']))
    torch.save(fb_tensor, str(fb_path))
    # Save waveform tensors
    wav_tensor = torch.tensor(group['waveform'].to_list())
    wav_path = tensor_path.joinpath(''.join(['waveforms_',name,'.pt']))
    torch.save(wav_tensor, str(wav_path))
    # Save label tensors
    label_tensor = torch.tensor(lb.transform(group['language']))
    label_path = tensor_path.joinpath(''.join(['labels_',name,'.pt']))
    torch.save(label_tensor, str(label_path))
    print("Saved tensors for", name, 'time elapsed:', 
          round((time()-start_time)/60, 2), "minutes")