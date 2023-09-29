# BERT-geoparser - Geolocation of textual data
In this project we develop a model for geolocating textual data. The developed model is based on a BERT language model, trained on the task of identifying target and incidental locations in text. Given a textual input, we define a *target* location as any location directly related to the location of the writer (or the location that is the main subject of the text), and an *incidental* location as other locations which are mentioned, but do not relate to the location of writer. For example, consider the sentence:

<p style="text-align: center;"> "I am in Glasgow Airport waiting for a plane to Singapore."</p>

In this sentence, *Glasgow Airport* is the target location and *Singapore* is an incidental location, unrelated to the true location of the writer. This project aims to construct a language model that is able to (a) identify locations in text (the easy bit) and (b) use the grammatical structure of the sentence to assign locations as either *target* or *incidental*.

To do this, we take the following steps outlined in table 1:

|Step|Process|Output|Data Required|Notebooks|
|----|-------|------|-------------|---------|
|1.|Training a BERT model on a Named Entity Recognition (NER) dataset, with tags including location, organisation, name and geo-political entity.|BERT-Model trained on NER data.| [NER-Dataset](https://www.kaggle.com/datasets/namanj27/ner-dataset).|[Notebook 1](notebooks/NB1_training_an_NER_Bert_model.ipynb).|
|2.  |NER-parsing of a geo-coded textual dataset to identify location tags|NER tagged geocoded textual data.|Geocoded textual data. Initially using the [Yelp review dataset](https://www.yelp.com/dataset) athough this is not ideal.|[Notebook 1](notebooks/NB1_training_an_NER_Bert_model.ipynb).|
|3.  |Retagging of geocoded textual data, based of proximity of NER identified locations to "true" location of text.|Dataset tagged with *target* and *incidental* locations.|Output from step 2.|[Notebook 2](notebooks/NB2_target_location_identification_with_BERT.ipynb).|
|4.  |Re-training the BERT model on the retagged dataset.|BERT-model trained on target/incidental location identification.|Output from step 3.|[Notebook 2](notebooks/NB2_target_location_identification_with_BERT.ipynb).|

Further model development and analysis is handled in notebook NB3. The two main external datasets used in model development are quite large and have not been included in this repository. Please follow the links in the table to download this data. 

The file structure used in the code requires the NER dataser be placed in the directory `/data/training_step_1/` and the yelp review data (or other geocoded dataset) to be placed in `data/training_step_2/`. 

If a different geocoded dataset is used then it should follow the column headings used in the yelp data, namely `text` for the text strings and `coordinates` for the location data in (longitude, lattitude) format.

## Requirements
* python==3.9
* tensorflow==2.10
* jupyter
* pandas
* numpy
* geopy
* shapely
* tqdm
* sklearn
* pathlib
* transformers
* tokenizers

It is recomended to use a reasonably large CUDA enabled GPU for model training. Pre-trained models are from steps 1 and 4 are in the `/models/` directory.
