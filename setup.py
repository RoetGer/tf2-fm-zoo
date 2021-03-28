import os.path as osp
from setuptools import setup, find_packages


# read the contents of your README file
rep_folder = osp.abspath(osp.dirname(__file__))
with open(osp.join(rep_folder, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="tf2_fm_zoo",
    version="0.1.2",
    url="https://github.com/RoetGer/tf2-fm-zoo",
    packages=find_packages(include=["fm_zoo"]),
    long_description=long_description, 
    long_description_content_type="text/markdown"
)