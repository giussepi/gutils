# G-utils

## Installation

### Package installation

Add to your requirements file:

``` bash
# use the latest version

gutils @ https://github.com/giussepi/gutils/tarball/master

# or use a specific release (format 1)

gutils @ https://github.com/giussepi/gutils/archive/refs/tags/v1.2.1.tar.gz

# or use a specific release (format 2)

gutils @ git+https://github.com/giussepi/gutils.git@v1.2.1
```

or install it directly:

```bash
pip install git+git://github.com/giussepi/gutils.git --use-feature=2020-resolver --no-cache-dir

# or

pip install https://github.com/giussepi/gutils/tarball/master  --use-feature=2020-resolver --no-cache-dir
```

### Development installation

1. Clone this repository.
2. Create your local settings:

	```bash
	cp settings.py.template settings.py
	```
3. Ensure QUICK_TEST is to False at `settings.py`
4. Modify or add new modules/features with their respective tests
5. Get the test datasets by running:

	```bash
	chmod +x get_test_datasets.sh
	./get_test_datasets.sh`
	```

6. Execute all the tests.

	```bash
	chmod +x run_tests.sh
	./run_tests.sh
	```

7. If all the tests pass, commit your changes.


A few of our tests employs two cases from the  **NIH-TCIA CT Pancreas benchmark (CT-82)** [^1] [^2] [^3]


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


## TODO
- [ ] Write more tests


[^1]: Holger R. Roth, Amal Farag, Evrim B. Turkbey, Le Lu, Jiamin Liu, and Ronald M. Summers. (2016). Data From Pancreas-CT. The Cancer Imaging Archive. [https://doi.org/10.7937/K9/TCIA.2016.tNB1kqBU](https://doi.org/10.7937/K9/TCIA.2016.tNB1kqBU)
[^2]: Roth HR, Lu L, Farag A, Shin H-C, Liu J, Turkbey EB, Summers RM. DeepOrgan: Multi-level Deep Convolutional Networks for Automated Pancreas Segmentation. N. Navab et al. (Eds.): MICCAI 2015, Part I, LNCS 9349, pp. 556–564, 2015.  ([paper](http://arxiv.org/pdf/1506.06448.pdf))
[^3]: Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L, Prior F. The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository, Journal of Digital Imaging, Volume 26, Number 6, December, 2013, pp 1045-1057. DOI: [https://doi.org/10.1007/s10278-013-9622-7](https://doi.org/10.1007/s10278-013-9622-7)
