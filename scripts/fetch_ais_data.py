import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import requests
from zipfile import ZipFile
from tqdm import tqdm

DATA_DIR = "./data/noaa_ais"
os.makedirs(DATA_DIR, exist_ok=True)

# 2024 sample (Zone 10 - San Francisco / West Coast)
# https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2024/AIS_2024_01_01.zip
# I'll just provide a direct link for a single day sample.

def download_ais_sample(day="2024_01_01"):
    url = f"https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2024/AIS_{day}.zip"
    target_path = os.path.join(DATA_DIR, f"AIS_{day}.zip")
    
    if not os.path.exists(target_path):
        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(target_path, 'wb') as file, tqdm(
            desc=f"AIS_{day}.zip",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    
    # Extract
    with ZipFile(target_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
        print(f"Extracted to {DATA_DIR}")

if __name__ == "__main__":
    download_ais_sample()
