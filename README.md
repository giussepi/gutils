
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


## Tools available:
### gutils/context_managers.py
- tqdm_joblib

### gutils/datasets/hdf5/exporters.py
- Images2HDF5

### gutils/datasets/hdf5/writers.py
- HDF5DatasetWriter

### gutils/datasets/utils
- calculate_train_val_percentages
- resize_dataset
- TrainValTestSplit

### gutils/decorators.py
- timing

### gutils/exceptions/common.py
- ExclusiveArguments

### gutils/files.py
- get_filename_and_extension
- clean_json_filename

### gutils/folders.py
- remove_folder
- clean_create_folder

### gutils/images/files.py
- list_images

### gutils/images/postprocessing
- RemoveBG

### gutils/images/preprocessing
- AspectAwarePreprocessor

### gutils/images/augmentation.py
- ImageAugmentationProcessor

### gutils/images/processing.py
- get_slices_coords
- get_patches

### gutils/mock.py
- notqdm

### gutils/numpy_/images.py
- ZeroPadding

### gutils/numpy_/numpy_.py
- colnorms_squared_new
- get_unique_rows
- normcols
- scale_using_general_min_max_values
- split_numpy_array
- LabelMatrixManager

### gutils/plots
- BoxPlot
- plot_color_table

### gutils/utils.py
- get_random_string


## TODO
- [ ] Write more tests
