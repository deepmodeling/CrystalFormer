from setuptools import setup, find_packages

setup(
    name = 'crystalformer',
    version = '0.3',
    keywords='Crystal Generation',
    description = 'CrystalFormer is a transformer-based autoregressive model specifically designed for space group-controlled generation of crystalline materials.',
    license = 'Apache License',
    url = 'https://github.com/deepmodeling/CrystalFormer',
    author = 'iopcompphys',
    author_email = 'zdcao@iphy.ac.cn, wanglei@iphy.ac.cn',
    packages = find_packages(),
    include_package_data = True,
    package_data={
        'crystalformer': ['data/*.csv'],
    },
    platforms = 'any',
    install_requires = [],
)