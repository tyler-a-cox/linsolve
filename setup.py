from setuptools import setup
import glob
import os.path as op
from os import listdir
from linsolve import version
import json

data = [version.git_origin, version.git_hash, version.git_description, version.git_branch]
with open(op.join('linsolve', 'GIT_INFO'), 'w') as outfile:
    json.dump(data, outfile)
import glob

setup_args = {
    'name': 'linsolve',
    'author': 'HERA Team',
    'url': 'https://github.com/HERA-Team/linsolve',
    'license': 'BSD',
    'description': 'high-level tools for linearizing and solving systems of equations',
    'package_dir' : {'linsolve':'linsolve'},
    'packages' : ['linsolve'],
    'version': version.version,
    'include_package_data': True,
    'install_requires': ['numpy>=1.10', 'scipy'],
    'classifiers': ['Development Status :: 5 - Production/Stable',
                    'Intended Audience :: Science/Research',
                    'License :: OSI Approved :: BSD License',
                    'Programming Language :: Python :: 2.7',
                    'Topic :: Scientific/Engineering :: Mathematics'],
    'keywords': 'linear equations optimal estimation',
}

if __name__ == '__main__':
    apply(setup, (), setup_args)
