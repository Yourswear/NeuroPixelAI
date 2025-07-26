import setuptools
from os import path
this_directory = path.abspath(path.dirname(__file__))

install_deps = [
    'datetime==5.4',
    'tqdm==4.66.2',
    'fastremap==1.14.1',
    'scikit-learn==1.2.2',#==1.2.2
    'numpy==1.18.5',#1.18.5
    'scipy==1.4.1',
    'torch',#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    'tifffile',
    'paramiko==3.4.0',
    'pillow==9.1',#9.1
    'PyQt5',
    'pyqt5-sip==12.12.2',
    'scikit-image==0.17.2',#0.17.2
    'opencv-python==4.9.0.80',
    'pyqtgraph==0.12.4',#==0.12.4
    'ruamel.yaml==0.18.6',
    'tensorflow==2.3.0',#2.3.0
    'keras==2.3.1',
    'natsort==8.4.0',
    'numba==0.56.4',
    'matplotlib==3.5.0',#==3.5
    'protobuf==3.20.1',#==3.20.1
    'csaps==1.1.0',
    'einops',
    'timm==0.5.4',
]


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="neuroai",
    version="0.4.7",
    author="HuJiahao",
    author_email="dadadadasukede@gmail.com",
    description="A package for calcium image processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=" ",
    install_requires=install_deps,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    python_requires='>=3.8'
)