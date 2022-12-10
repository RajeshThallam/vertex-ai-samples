
import os
from setuptools import find_packages, setup

requirements_file = 'requirements.txt'
REQUIRED_PACKAGES = []

if os.path.exists(requirements_file):
    REQUIRED_PACKAGES = [line for line in open('requirements.txt').read().splitlines()]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Vertex AI | Training | PyTorch Efficient Training | Python Package'
)
