# G-utils

## Installation

Add to your requirements file:

`gutils @ https://github.com/giussepi/gutils/tarball/master`

or run

```bash
pip install git+git://github.com/giussepi/gutils.git --use-feature=2020-resolver --no-cache-dir

# or

pip install https://github.com/giussepi/gutils/tarball/master  --use-feature=2020-resolver --no-cache-dir
```

## Usage

Explore the modules, load snippets and have fun! :blush::bowtie::nerd_face: E.g.:


```python
from gutils.decorators import timing

@timing
def my_function(*args, **kwargs):
    pass
```
