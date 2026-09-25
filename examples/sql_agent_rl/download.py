"""Fetch the pinned public Spider archive with checksum and safe extraction."""
import argparse
import hashlib
from pathlib import Path
import shutil
import urllib.request
import zipfile

REVISION = '6232cc3fad6d54c62b3ba23a364083a98ff36a17'
SHA256 = '5ddff97bb1d421282c593e8d30ce0ce107270f4dd4a21d60eba4bf287d5956b1'


def download(destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination/'spider.zip'
    if not archive.exists():
        temporary = archive.with_suffix('.part')
        with urllib.request.urlopen(f'https://huggingface.co/datasets/xlangai/spider/resolve/{REVISION}/data/spider.zip', timeout=120) as src, temporary.open('wb') as dst:
            shutil.copyfileobj(src, dst)
        temporary.rename(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise ValueError('Spider archive checksum mismatch')
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not (destination/name).resolve().is_relative_to(destination):
                raise ValueError('unsafe archive member')
        z.extractall(destination)
    print(destination/'spider')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('destination')
    download(parser.parse_args().destination)
