## Build the wheel
First generate the c files from all pyx files in napiod/. Then,

```
$ pip install -q build
$ python -m build
```


## Install in development mode
See [development-mode](https://setuptools.pypa.io/en/latest/userguide/quickstart.html#development-mode)

```
$ pip install --upgrade --force-reinstall --editable <path-to-mpoints-root>
```

