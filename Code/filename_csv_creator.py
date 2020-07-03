from pathlib2 import Path
import pandas as pd

# Change this and filename for different datasets
path = Path().joinpath('Common_Voice')
# Get all audio files
files = [x for x in path.glob('**/*') if x.suffix in ['.mp3','.wav','.flac']]

df = pd.DataFrame({'filename':files})
df.to_csv('common_voice_files.csv', index=False)