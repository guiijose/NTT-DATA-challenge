# NTT-DATA-challenge

Data processing pipeline for the Machine Learning merit prize. This repository extracts TF-IDF features from raw text data and prepares train/validation/test sets for downstream models.

## Project Structure

```bash
.
├── README.md
├── data/
│   ├── processed/              # Processed datasets (CSV files)
│   └── raw/                    # Raw datasets
├── requirements.txt            # Python dependencies
└── src/
    ├── main.ipynb              # Notebook for model training and analysis
    └── tfidf_extractor.py      # Data processing script
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
