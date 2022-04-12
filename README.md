# Sparse Constellations Source Code and Dataset
Welcome to the Sparse Constellations Dataset and Code Repository. Here you'll find an example API to access the dataset using python and numpy.

# Overview
This dataset contains about 640K annotated instances, segmented by day, room, and location.
Each data instance contains roughly 1207 features. It also contains data for in-room events, as well as synchronized data streams from all other sensors.  

# System Requirements
python3, numpy, and pickle

# Dataset Files
The dataset files are hosted on dropbox (too big for Github). Download them here:
https://www.dropbox.com/sh/bxx86a79we730ic/AAD1ENqMMrKgAsXBXkgF-d-Ra?dl=0

# Code Usage
The data folder contains python serialized classes of all collected data. Make sure you download the files above, and save it as '/data'.

sample_loop.py lists the neccessary routines for acessing the data.

The basic steps for accessing the raw numpy arrays are as follows:

1. Unpickle the data file for a room using:
```python
room_dataset = load_data('data.pklz')
```
2. To access data instances for that room, use:
```python
room_dataset['Day1']['data'] for 'Day1' data. Data goes up to 'Day7'
```
3. To access data labels, use:
```python
data = room_dataset['Day1']['labels']
```

