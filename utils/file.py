# utils/file.py

import requests
import zipfile
import tarfile
from pathlib import Path


def download_file(url: str, filename: str) -> None:
 
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded file: {filename}")
    except requests.RequestException as e:
        print(f"[ERROR] Failed to download {url}: {e}")


def extract_tar_gz(tar_gz_path: str, dest_path: str) -> None:
  
    try:
        with tarfile.open(tar_gz_path, 'r:gz') as tar:
            tar.extractall(path=dest_path)
        print(f"Successfully extracted {tar_gz_path} to {dest_path}")
    except (FileNotFoundError, tarfile.TarError) as e:
        print(f"[ERROR] Failed to extract {tar_gz_path}: {e}")


def extract_tgz(file_path: str, extract_path: str) -> None:
   
    try:
        with tarfile.open(file_path, 'r:gz') as tar:
            tar.extractall(path=extract_path)
        print(f"tgz file {file_path} has been extracted to {extract_path}")
    except FileNotFoundError:
        print(f"[ERROR] File {file_path} does not exist.")
    except tarfile.ReadError:
        print(f"[ERROR] {file_path} is not a valid tgz file.")


def unzip_file(zip_filepath: str, dest_path: str) -> None:
 
    try:
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(dest_path)
        print(f"Successfully extracted {zip_filepath} to {dest_path}")
    except FileNotFoundError:
        print(f"[ERROR] File {zip_filepath} does not exist.")
    except zipfile.BadZipFile:
        print(f"[ERROR] {zip_filepath} is not a valid zip file.")
    except Exception as e:
        print(f"[ERROR] Failed to extract {zip_filepath}: {e}")
