
import os
import requests
import zipfile
from pathlib import Path

# setup path
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# download if, image folder doesn't exists
if image_path.is_dir():
  print(f"{image_path} directory already exists...skipping download")
else:
  print(f"{image_path} does not exists...creating one")
  image_path.mkdir(parents = True, exist_ok = True)


# download zip file from daniel github
with open(data_path/ "pizza_steak_sushi.zip", 'wb') as f:
  request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
  print('Downloading....the github zip file')
  f.write(request.content)

# unzip the data
with zipfile.ZipFile(data_path/'pizza_steak_sushi.zip', 'r') as zip_ref:
  print('Unzipping the zip file')
  zip_ref.extractall(image_path)

# remove the zip file
os.remove(data_path/'pizza_steak_sushi.zip')
