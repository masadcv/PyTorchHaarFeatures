import setuptools

def readme():
    with open('README.md') as f:
        return f.read()

setuptools.setup(name='torchhaarfeatures',
      version='0.0.2',
      description='Haar-like features using PyTorch',
      long_description=readme(),
      long_description_content_type="text/markdown",
      classifiers=[
        'Operating System :: OS Independent',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
      ],
      keywords='haar-like 2d 3d medical features hand-crafted',
      url='http://github.com/masadcv/PyTorchHaarFeatures',
      author='Muhammad Asad',
      author_email='muhammad.asad@kcl.ac.uk',
      license='BSD-3-Clause',
      packages=['torchhaarfeatures'],
      install_requires=[
          'torch',
      ],
      zip_safe=False)
