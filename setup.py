import glob

__version__ = '0.0.1'

setup_args = {
    'name': 'linsolve',
    'author': 'Aaron Parsons and Joshua Dillon',
    'author_email': 'aparsons at berkeley.edu',
    'license': 'GPL',
    'package_dir' : {'linsolve':'src'},
    'packages' : ['linsolve'],
    'version': __version__,
}

if __name__ == '__main__':
    from distutils.core import setup
    apply(setup, (), setup_args)
