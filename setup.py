from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirments(file_path: str)->List[str]:
    requirments = []
    with open('requirements.txt') as file_obj:
        requirments = file_obj.readlines()
        [req.replace('\n', "") for req in requirments]

        if HYPEN_E_DOT in requirments:
            requirments.remove(HYPEN_E_DOT)
    return requirments

setup(
    name = "Cardamom Project",
    version="0.0.1",
    author="Roshan Chhetri",
    author_email="roshanchhetri931@gmail.com",\
    packages=find_packages(),
    install_requires = get_requirments('requirements.txt')
)