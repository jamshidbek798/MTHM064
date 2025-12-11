"""
This script trains Supervised Autoencoder (SAE) models for the CRISPRi_targeted dataset in a cloud environment.

Description:
------------
The script automates the training and evaluation of supervised autoencoders for large-scale single-cell CRISPRn
perturbation data. It uses previously optimized hyperparameter configurations to train models across multiple
latent dimensions and identifies perturbation effects (effect vs no effect) using Euclidean distance in the latent space.

Workflow:
---------
1. Loads preprocessed CRISPRi_targeted expression and metadata files.
2. Merges and prepares the data using `preprocess_data()`.
3. Loads pre-optimized model hyperparameters from pickle.
4. Performs multiple random train/test splits for robust evaluation.
5. Trains supervised autoencoders for multiple latent dimensions using early stopping.
6. Computes Euclidean distances between gRNA clusters and non-targeting controls.
7. Determines thresholds separating 'effect' and 'no effect' groups.
8. Evaluates clustering with Silhouette Scores for both train and test sets.
9. Saves all results as timestamped CSV files for downstream analysis.

Output:
-------
Each training run produces a CSV file in:
    CRISPRi_targeted_samples/CRISPRi_targeted_sample_<timestamp>.csv

The file contains:
    sample, dimension, method, y_label, target_gene, gRNA_simplified,
    effect_label, sil_score_train, sil_score_test
"""

# Import necessary libraries
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import silhouette_score
from optimisation import *

# Load expression data (log-transformed counts)
logcounts = pd.read_csv("/CRISPRi_targeted_upr_logcounts.csv")

# Load sample metadata (gRNA, target gene, etc.)
meta_data = pd.read_csv("/meta_CRISPRi_targeted.csv")

# Merge and clean data for modeling
data = preprocess_data(logcounts, meta_data)

# Load pre-optimized model configurations
import pickle
with open("Training Config/results_history_configs_CRISPRi_trageted.pkl", "rb") as f:
    results_history_configs = pickle.load(f)

# Define training settings
EPOCHS = 300
BATCH_SIZE = 256

# Create an empty DataFrame to store results
effect_results_by_dim = pd.DataFrame(
    columns=["sample", "dimension", "method", "y_label", "target_gene", "gRNA_simplified", "effect_label", "sil_score_test", "sil_score_train"]
)

# Define latent dimensions for model training
latent_dim = [2, 3, 4, 5, 8, 12, 18, 27, 41, 62, 72, 92]

# Number of samples to train
total_samples = 30

# Loop through each sample iteration
for sample in range(total_samples):
    # Split data into train and test sets
    X_train, X_test, target_gene_train, target_gene_test, gRNA_train, gRNA_test = (
        get_train_test_grna_sample(data))

    # Loop over label types (gRNA-level and target gene-level)
    for y_label in ["gRNA", "target_gene"]:

        le = LabelEncoder()
        # Fit encoder on training labels only
        y_train = le.fit_transform(target_gene_train if y_label == "target_gene" else gRNA_train)
        # Transform test labels using same encoder
        y_test = le.transform(target_gene_test if y_label == "target_gene" else gRNA_test)

        num_classes = len(np.unique(y_train))
        best_models_euclidean = {}
        silhouette_score_history_euclidean = []

        # Loop through each latent dimension configuration
        for dim in latent_dim:
            config = results_history_configs[y_label]["euclidean"]['best_models'][dim]['config']

            print(f"Training: sample={sample}, latent_dim={dim}")

            # Build supervised autoencoder model
            autoencoder, encoder = supervised_autoencoder(
                input_dim=X_train.shape[1],
                latent_dim=dim,
                units=config["units"],
                dropout=config["dropout"],
                learning_rate=config["learning_rate"],
                optimizer=config["optimizer"],
                activation=config["activation"],
                num_classes=num_classes,
                y_train=y_train
            )

            # Define early stopping callback
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            # Train the model
            history = autoencoder.fit(
                X_train,
                {"reconstruction": X_train, "classification": y_train},
                validation_split=0.2,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=0,
                callbacks=[early_stop]
            )

            # Encode train/test data into latent representations
            X_test_latent = encoder.predict(X_test)
            X_train_latent = encoder.predict(X_train)

            # Compute Euclidean distances between gRNA groups and control
            distance_euclidean = compute_distance(X_train_latent, gRNA_train, method="euclidean")
            threshold_euclidean = compute_threshold_and_plot_hist(distance_euclidean, method="euclidean")

            # Label gRNAs as 'effect' or 'no effect' based on threshold
            distance_euclidean["effect"] = np.where(distance_euclidean > threshold_euclidean, "effect", "no effect")
            effect_dict_euclidean = dict(zip(distance_euclidean.index, distance_euclidean["effect"]))
            true_labels_effect_test_euclidean = np.array([effect_dict_euclidean.get(g) for g in gRNA_test])
            true_labels_effect_train_euclidean = np.array([effect_dict_euclidean.get(g) for g in gRNA_train])

            # Evaluate model using Silhouette Scores
            sil_score_test_euclidean = silhouette_score(X_test_latent, true_labels_effect_test_euclidean, metric='euclidean')
            sil_score_train_euclidean = silhouette_score(X_train_latent, true_labels_effect_train_euclidean, metric='euclidean')
            print("Test set Silhouette score:", sil_score_test_euclidean)
            print("Train set Silhouette score:", sil_score_train_euclidean)

            # Map effect labels to dataset
            data_copy = data.copy()
            data_copy["effect"] = data_copy["gRNA_simplified"].map(effect_dict_euclidean)

            # Prepare result DataFrame for current model
            df = data_copy[["target_gene", "gRNA_simplified", "effect"]].drop_duplicates()
            df["effect_label"] = (df["effect"] == "effect").astype(int)

            # Add metadata columns
            df["sample"] = sample
            df["dimension"] = dim
            df["method"] = "euclidean"
            df["y_label"] = y_label
            df["sil_score_train"] = sil_score_train_euclidean
            df["sil_score_test"] = sil_score_test_euclidean

            # Append current results to global DataFrame
            effect_results_by_dim = pd.concat([effect_results_by_dim, df], ignore_index=True)

# Save all training results to CSV
import os
import time
os.makedirs("CRISPRi_targeted_samples", exist_ok=True)
effect_results_by_dim.to_csv(f"CRISPRi_targeted_samples/CRISPRi_targeted_sample_{int(time.time())}.csv", index=False)