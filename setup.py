from setuptools import find_packages
from setuptools import setup

setup(name='basars_addons',
      version='0.0.4',
      description='TensorFlow addons that help to build ML models for Basars',
      url='https://github.com/Basars/basars-addons.git',
      author='OrigamiDream',
      packages=find_packages(),
      zip_safe=False,
      include_package_data=True,
      install_requires=[
          'tensorflow>=2.0',
          'numpy',
      ],
      extra_require={
          'tests': ['pytest', 'opencv-python']
      })
