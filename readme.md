# Using Transformer Based Models to Identify Spatial Relationships Between Toponyms

## Requirements
The following environment was used to train the models and run the analysis as described in the paper.

* python==3.10.13
* numpy==1.26.0
* pandas==2.0.3
* scikit-learn==1.3.0
* transformers==4.32.1
* tokenizers==0.13.0
* tensorflow==2.10.0
* shapely==2.0.1
* geopy==2.4.0
* wikipedia==1.4.0
* matplotlib
* tqdm

The models used in this work are very large and can take a long time to train. We highly recommend training on CUDA enabled GPUs.

The software has been run successfully on Windows 10 and Linux CentOS 7 machines.

## Package contents

This repository includes the BERT_geoparser package used to train and test a TopoBERT model on relationally tagged data. The scripts included in the package are:
* `data.py` - used to process data to input into models.
* `tokenizer.py` - a wrapper around python's `tokenizers` package, used to convert words into tokens. 
* `model.py` - builds, trains and tests the TopoBERT model.
* `gazetteer.py` - handles queries to the OpenStreetMaps API.
* `relational_retagger.py` - assigns tags to toponyms according to the spatial relationships defined in the paper.
* `analysis.py` - a few functions for assessing test results.
* `utils.py` - a couple of utility functions.

## Notebooks

The analysis pipeline is given in four notebooks, found in the `/notebooks/` directory. These should be run in order, however be aware that some take a significant time to run - particularly the collection of data from Wikipedia in NB2 and the relational retagging in NB3 The data produced at each step has been made available in the `\data\` repository, so some of the steps can be skipped to save time. The notebooks are

* `NB1_training_TopotBERT_on_NER_data.ipynb`
* `NB2_extracting_wikipedia_data.ipynb`
* `NB3_relational_retagging.ipynb`
* `NB4_training_on_relationally_retagged_data.ipynb`

## Data
We have included the data produced in each step of the analysis pipeline. Data is stored in the `\data\` directory, and organised into folders according to the notebook in which they are used and/or created. Data dictionaries are included in each folder. The dataset `wiki_places_reviewed.csv` includes the human annotation used to test the model. This can be overwritten by running NB3, so please take care to back up the old data if reproducing the results. The `geopackage` files used to generate the map in figure 1 have not been included.

## Results
All the results included in the paper have been included in the notebooks or in the `/results/` folder. The map used in figure 1 is included in `/results/` however since the map is primarily illustrative and unrelated to any results, the steps for reproduction have not been included.

## Models
We have included the final trained model as a hdf5 files, however due to repository size limitations intermediary models have not been included. The full model is 3gb in size.

## License
GNU General Public License (GPL) 2.0