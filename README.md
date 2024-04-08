IDS-System-Project
==============================

# Intrusion Detection System Python Project

This project proposes the development of a model for Intrusion Detection in Computer Networks. 

## Requirements and Uses

### Prerequisites:
To clone and run the project, the following software tools are needed:

* [Git](https://git-scm.com/downloads) 
* [Python3](https://www.python.org/downloads/)
* [Pip3](https://bootstrap.pypa.io/get-pip.py)

This project requieres the libraries described in the requirements.txt file

### Source code:
First, it is highly recommended to fork the project from your personal GitHub account. Then, once the fork is created, download the sources with the command:

`git clone https://github.com/yourUser/ids_experimenter.git`
 
 or

`git clone git@github.com:yourUser/ids_experimenter.git`


* It can also be downloaded as a .zip file and unzipped.


### Run and Uses:
To run follow the next steps:

1. Install the virtual environment: 
  * Intalling the environment:`python3 -m pip install virtualenv`
  * Verify instalation: `virtualenv --version`

2. Create a virtual environment and activate it:
  * On Windows: `py -m venv .venv && .venv\Scripts\activate`
  * On linux and Mac: `python3 -m venv .venv && source .venv/bin/activate`

3. Install the libraries contained in the "requirements.txt" file: `pip install -r requirements.txt`

## Development

### Important links:
* Official source code repo: -
* Download releases: -

### Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── Datasets         <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
