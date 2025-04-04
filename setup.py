from setuptools import setup, find_packages

setup(
    name = 'crystalformer',
    version = '0.4.2',
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
    platforms = 'linux',
    install_requires = [],
    entry_points = {
        'console_scripts': [
            "train_ppo=crystalformer.cli.train_ppo:main",
            "train_dpo=crystalformer.cli.train_dpo:main",
            "classifier=crystalformer.cli.classifier:main",
            "cond_gen=crystalformer.cli.cond_gen:main",
            "dataset=crystalformer.cli.dataset:main",
            "spg_sample=crystalformer.cli.spg_sample:main",
        ]
    }
)
