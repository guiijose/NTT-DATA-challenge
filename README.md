# NTT-DATA-challenge

Data processing pipeline for the Machine Learning merit prize. This repository extracts TF-IDF features from raw text data and prepares train/validation/test sets for downstream models.

## Project Structure

```bash
.
├── README.md
├── data
│   ├── processed
│   │   ├── test_features.csv           # TF-IDF encoded datasets
│   │   ├── train_features.csv
│   │   └── validation_features.csv
│   ├── raw
│   │   ├── test.csv                    # Raw datasets
│   │   ├── train.csv
│   │   └── validation.csv
│   └── vectorizer.pkl                  # Serialized vectorizer object
├── models                              # Serialized model objects
│   ├── decision_tree.pkl
│   ├── gaussian_nb.pkl
│   ├── lr_l1.pkl
│   ├── lr_l2.pkl
│   └── mlp.pkl
├── requirements.txt
└── src                                 # Notebooks
    ├── clustering.ipynb
    ├── model_evaluation.ipynb
    ├── model_interpretation.ipynb
    ├── model_training.ipynb
    └── tfidf_extractor.py              # TF-DF encoder scripts

6 directories, 19 files

```

## Setup

### 1. Clone Repository

```bash
git clone git@github.com:guiijose/NTT-DATA-challenge.git
cd NTT-DATA-challenge
```

### 2. Create Virtual Environment

```bash
python -m venv env
```

### 3. Activate Virtual Environment

**Linux/MacOS:**
```bash
source env/bin/activate
```

**Windows:**
```bash
env\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Data Preprocessing

To preprocess the raw data and extract TF-IDF features, run:

```bash
python src/tfidf_extractor.py \
  --train_path data/raw/train.csv \
  --test_path data/raw/test.csv \
  --validation_path data/raw/validation.csv \
  --output_folder data/processed
```

or you can run the necessary functions in the `model_training.ipynb` notebook.