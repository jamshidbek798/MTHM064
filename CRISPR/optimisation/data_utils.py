"""
Functions to preprocess CRISPR gene expression data and generate train/test splits
with gRNA and target gene labels.

Includes:
- preprocess_data: merges logcounts with metadata and standardizes gRNA/target gene labels.
- get_train_test_grna: produces a stratified train/test split and scales features.
- get_train_test_grna_sample: same as above but with a different test size and random seed.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def preprocess_data(logcounts, meta_data):
    """
    Merge logcounts and metadata, simplify gRNA labels, and fix control gene inconsistencies.

    Parameters
    ----------
    logcounts : pd.DataFrame
        Gene expression logcounts with cells as columns.
    meta_data : pd.DataFrame
        Metadata including gRNA and target gene info.

    Returns
    -------
    pd.DataFrame
        Merged and cleaned dataset.
    """
    # Match sample names: replace "-" with "." for merging
    meta_data.index = meta_data.index.str.replace("-", ".", regex=False)
    # Merge gene expression and metadata
    data = pd.merge(logcounts.T, meta_data, how="left", left_index=True, right_index=True)

    # Simplify gRNA labels
    data["gRNA_simplified"] = data["gRNA"].apply(
        lambda g: "Non-Targeting" if g.startswith("Non-Targeting") else g
    )
    # Fix inconsistent control label
    data["target_gene"] = data["target_gene"].replace("GSH-Rogi1", "Control")
    return data


def get_train_test_grna(data):
    """
    Generate stratified train/test split for gRNA and target gene labels with feature scaling.

    Parameters
    ----------
    data : pd.DataFrame
        Preprocessed dataset.

    Returns
    -------
    tuple
        X_train, X_test, target_gene_train, target_gene_test, gRNA_train, gRNA_test
    """
    X_all = np.array(data.drop(columns=["gRNA", "target_gene", "gRNA_simplified"])).astype(np.float32)
    target_gene_all = np.array(data["target_gene"])
    gRNA_all = np.array(data["gRNA_simplified"])

    # Stratified train-test split
    X_train, X_test, target_gene_train, target_gene_test, gRNA_train, gRNA_test = train_test_split(
        X_all, target_gene_all, gRNA_all, test_size=0.2, random_state=42, stratify=gRNA_all
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, target_gene_train, target_gene_test, gRNA_train, gRNA_test


def get_train_test_grna_sample(data):
    """
    Generate a 50/50 stratified train/test split for gRNA and target gene labels
    with feature scaling. Random seed is not fixed.

    Parameters
    ----------
    data : pd.DataFrame
        Preprocessed dataset.

    Returns
    -------
    tuple
        X_train, X_test, target_gene_train, target_gene_test, gRNA_train, gRNA_test
    """
    X_all = np.array(data.drop(columns=["gRNA", "target_gene", "gRNA_simplified"])).astype(np.float32)
    target_gene_all = np.array(data["target_gene"])
    gRNA_all = np.array(data["gRNA_simplified"])

    X_train, X_test, target_gene_train, target_gene_test, gRNA_train, gRNA_test = train_test_split(
        X_all, target_gene_all, gRNA_all, test_size=0.5, random_state=None, stratify=gRNA_all
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, target_gene_train, target_gene_test, gRNA_train, gRNA_test