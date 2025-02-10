import gdown
import os
import zipfile

# URL of the Google Drive folder
folder_url = 'https://drive.google.com/drive/folders/1OwfXDwuQazuEYXPb_B3nLt6uOY4IKxiq'

# Extract the folder ID from the URL
folder_id = folder_url.split('/')[-1]

# Output file name for the downloaded zip
output = 'dataset.zip'

# Download the folder as a zip file
gdown.download_folder(f'https://drive.google.com/drive/folders/{folder_id}', output=output, quiet=False, use_cookies=False)

# Unzip the downloaded file
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall('datasets')

# Remove the zip file after extraction
os.remove(output)

print('Dataset downloaded and extracted successfully.')
