### Clone the repository
0. Clone the rpository and its submodules:
```
$ git clone --recursive <napiod-url> 
```

### Create a conda envirnoment where NAPIOD will be installed

Create a conda environment where we install `napiod` and additional packages. 
We use the file `napiod.yml` to define such an environment. 

1. Install the conda environment:
```
$ conda env create -f napiod.yml
```

2. Activate the environment:
```
$ conda activate napiod
```

3. [Optional] Install additional libraries (useful in the tests and examples):
```
$ conda install pytest
$ conda install seaborn
```


### Build and install NAPIOD
Navigate to the root directory of `napiod`.
Make sure that the environment `napiod` created in Step 1 is still activated. This is the same environment where you installed `mpoints` in Step 5. 

4. Generate the c code and build `napiod`:
```
$ cd napiod
$ python setup.py build_ext --inplace
$ cd ..
$ python -m build --sdist
```

5. The previous step (Step 6.) has generated a tar file in the riectory `dist/`. Use this file to install `napiod`:
```
$ pip install --force-reinstall <path-to-napiod-tar-file>

```

### [Optional] Run tests with pytest
```
$ pytest tests/*.py
```

### [Optional] Run examples
```
$ python examples/market_regimes.py
```




