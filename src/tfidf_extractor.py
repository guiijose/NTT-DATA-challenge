import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse
from argparse import ArgumentParser
import os

def prGreen(text : str) -> None:
    """
    Print text in green color in the terminal.

    Parameters:
        text (str): The text to be printed.
    """
    print(f"\033[92m{text}\033[0m")
    
def load_and_vectorize(
        train_path : str,
        test_path : str | None = None,
        validation_path : str | None = None,
        text_col : str = "Text",
        label_col : str = "Label",
        max_features : int = 5000,
        min_df : int = 10,
        max_df : float = 0.9,
        smooth_idf : bool = True
    ) -> tuple:
    """
    Load CSVs, extract TF-IDF features from the training set, and transform test/validation sets.

    Parameters:
        train_path (str): Path to the training CSV file.
        test_path (str|None): Path to the test CSV file (or None).
        validation_path (str|None): Path to the validation CSV file (or None).
        text_col (str): Name of the column containing document text.
        label_col (str): Name of the column containing labels.
        max_features (int): Max number of TF-IDF features.
        min_df (int): Minimum number of documents a term must appear in.
        max_df (float): Maximum proportion of documents a term can appear in.
        smooth_idf (bool): Apply IDF smoothing.

    Returns:
        X_train (sparse matrix): TF-IDF feature matrix for training.
        y_train (pd.Series): Training labels.
        X_test (sparse matrix|None): TF-IDF feature matrix for test (None if test_path is None).
        y_test (pd.Series|None): Test labels (None if test_path is None).
        X_val (sparse matrix|None): TF-IDF feature matrix for validation (None if validation_path is None).
        y_val (pd.Series|None): Validation labels (None if validation_path is None).
        vectorizer (TfidfVectorizer): Fitted vectorizer.
    """
    # Load train CSV
    df_train = pd.read_csv(train_path)
    train_texts = df_train[text_col].astype(str).tolist()
    y_train = df_train[label_col]

    # Initialize and fit vectorizer on training data
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        smooth_idf=smooth_idf
    )

    X_train = vectorizer.fit_transform(train_texts)

    # Helper to load and transform a dataset (returns (X, y) or (None, None))
    def _load_and_transform(path):
        if path is None:
            return None, None
        df = pd.read_csv(path)
        texts = df[text_col].astype(str).tolist()
        labels = df[label_col]
        X = vectorizer.transform(texts)
        return X, labels

    X_test, y_test = _load_and_transform(test_path)
    X_val, y_val = _load_and_transform(validation_path)

    return (X_train, y_train, X_test, y_test, X_val, y_val, vectorizer)

def write_to_file(file_path: str, X : 'scipy.sparse.csr_matrix', y : pd.Series) -> None:
    """
    Write a sparse matrix and labels to a file in CSV format.

    Parameters:
        file_path (str): Path to the output file.
        X (scipy.sparse.csr_matrix): Sparse matrix from TfidfVectorizer.
        y (pd.Series): Labels corresponding to the rows in X.
    """
    # Convert to dense array and then to DataFrame for easier saving
    dense_array = X.todense()
    df = pd.DataFrame(dense_array)
    df["Label"] = y.values
    df.to_csv(file_path, index=False)

if __name__ == "__main__":
    parser = ArgumentParser(description="Extract TF-IDF features and save to CSV.")
    parser.add_argument("--train_path", type=str, required=True, help="Path to the training CSV file")
    parser.add_argument("--test_path", type=str, help="Path to the test CSV file", default=None)
    parser.add_argument("--validation_path", type=str, help="Path to the validation CSV file", default=None)
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save the output CSV files")
    args = parser.parse_args()

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Load and vectorize all datasets
    X_train, y_train, X_test, y_test, X_val, y_val, vectorizer = load_and_vectorize(
        train_path=args.train_path,
        test_path=args.test_path,
        validation_path=args.validation_path
    )

    prGreen("TF-IDF feature extraction completed. Saving files...")

    # Save training features
    train_output = os.path.join(args.output_folder, "train_features.csv")
    write_to_file(train_output, X_train, y_train)

    prGreen("Training features saved.")

    # Save test features if available
    if X_test is not None:
        test_output = os.path.join(args.output_folder, "test_features.csv")
        write_to_file(test_output, X_test, y_test)
        prGreen("Test features saved.")
    else:
        prGreen("No test data provided; skipping test feature saving.")

    # Save validation features if available
    if X_val is not None:
        val_output = os.path.join(args.output_folder, "validation_features.csv")
        write_to_file(val_output, X_val, y_val)
        prGreen("Validation features saved.")
    else:
        prGreen("No validation data provided; skipping validation feature saving.")
