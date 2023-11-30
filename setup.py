# -*- encoding: utf-8 -*-
from pathlib import Path
import setuptools


def get_readme():
    root_dir = Path(__file__).resolve().parent
    readme_path = str(root_dir / 'README.md')
    print(readme_path)
    with open(readme_path, 'r', encoding='utf-8') as f:
        readme = f.read()
    return readme


MODULE_NAME = 'emo2vec'
VERSION_NUM = '0.0.1'

setuptools.setup(
    name=MODULE_NAME,
    version=VERSION_NUM,
    platforms="Any",
    url="git@github.com:ddlBoJack/emotion2vec.git",
    author="",
    author_email="",
    description="emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation",
    license='MIT',
    long_description=get_readme(),
    long_description_content_type='text/markdown',
    include_package_data=True,
    install_requires=["fairseq",
                      "numpy",
                      "soundfile",
                      "torch"
                      ],
    packages=["emo2vec"],
    keywords=[
        'ssl,emotion'
    ],
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    entry_points={"console_scripts": [
        "emo2vec = emo2vec.emo2vec_cli:main",
    ]},
)
