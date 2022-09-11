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

### gutils/images/images
- DICOM
- NIfTI
- ProNIfTI

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


## Development
After modifying or adding new modules with their respetive tests, make sure to do the following before committing any update:
1. Set QUICK_TEST to False at `gutils/settings.py`
2. Get the test datasets by running `./get_test_datasets.sh`
3. Execute all the tests `./run_tests.sh`.
4. Discard changes at the settings by executing `git checkout gutils/settings.py`
5. Commit your changes

A few of our tests employs two cases from the  **NIH-TCIA CT Pancreas benchmark (CT-82)** [^1] [^2] [^3]

## TODO
- [ ] Write more tests


[^1]: Holger R. Roth, Amal Farag, Evrim B. Turkbey, Le Lu, Jiamin Liu, and Ronald M. Summers. (2016). Data From Pancreas-CT. The Cancer Imaging Archive. [https://doi.org/10.7937/K9/TCIA.2016.tNB1kqBU](https://doi.org/10.7937/K9/TCIA.2016.tNB1kqBU)
[^2]: Roth HR, Lu L, Farag A, Shin H-C, Liu J, Turkbey EB, Summers RM. DeepOrgan: Multi-level Deep Convolutional Networks for Automated Pancreas Segmentation. N. Navab et al. (Eds.): MICCAI 2015, Part I, LNCS 9349, pp. 556–564, 2015.  ([paper](http://arxiv.org/pdf/1506.06448.pdf))
[^3]: Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L, Prior F. The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository, Journal of Digital Imaging, Volume 26, Number 6, December, 2013, pp 1045-1057. DOI: [https://doi.org/10.1007/s10278-013-9622-7](https://doi.org/10.1007/s10278-013-9622-7)
