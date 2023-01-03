# create a conda environment
conda create -p ./conda-env python=3.9

# activate env
conda activate ./conda-env

# upgrade pip
python3 -m pip install --upgrade pip

# install package from test pypi without dependencies
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps team20-ad

# install numpy from proper pypi
python3 -m pip install numpy

# how to use it
# python3
# >>> from team20ad.forwardAD import *
# >>> ad = ForwardAD({'x': 1, 'y': 1}, ['x**2 + y**2', 'exp(x + y)'])
# >>> ad()
