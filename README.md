# Bistable Perception in Conceptor Networks
This repository contains code for the paper "Bistable Perception in Conceptor Networks".

## Reproduction of results
To reproduce the results from the paper, do the following

* install [conda](https://docs.conda.io/en/latest/miniconda.html)
* create a virtual environment and install dependencies using


    $ conda env create -f env.yml

* activate the environment using


    $ conda activate bistable

* run the experiments

    
    $ cd binocular
    $ PYTHONPATH=.. python run.py
    

The results should show up in the directory `binocular/runs`. Some auxiliary plots will pop
up as well, but are not saved to disk.
