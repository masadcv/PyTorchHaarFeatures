import setuptools

def readme():
    with open('README.md') as f:
        return f.read()

setuptools.setup(name='pt_haarfeatures',
      version='0.1',
      description='Haar-like features using PyTorch',
      long_description=readme(),
      classifiers=[
        'Operating System :: OS Independent',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
      ],
      keywords='haar-like 2d 3d medical features hand-crafted',
      url='http://github.com/masadcv/PyTorchHaarFeatures',
      author='Muhammad Asad',
      author_email='masadcv@gmail.com',
      license='BSD-3-Clause',
      packages=['pt_haarfeatures'],
      install_requires=[
          'torch',
      ],
      zip_safe=False)