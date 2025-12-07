import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def preprocess_data(logcounts, meta_data):
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
    # Prepare numeric features and labels

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