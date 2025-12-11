"""
This script trains Supervised Autoencoder (SAE) models for the CRISPRi_whole dataset in a cloud environment.

Description:
------------
The script automates large-scale model training and effect detection using pre-optimized hyperparameter 
configurations. It evaluates clustering performance and identifies perturbations (effects) at the gRNA and 
target gene levels.

Workflow:
---------
1. Loads log-transformed gene expression and metadata files.
2. Merges and preprocesses data using 'preprocess_data()'.
3. Loads pre-optimized model configurations from a pickle file.
4. Splits data into train/test subsets for multiple random samples.
5. Trains a supervised autoencoder across multiple latent dimensions using early stopping.
6. Computes Euclidean distances in latent space to identify gRNAs with or without effects.
7. Evaluates latent space quality using Silhouette Scores for both train and test sets.
8. Saves all results to timestamped CSV files for downstream analysis.

Output:
-------
Each run generates a file in:
    CRISPRi_whole_samples/CRISPRi_whole_sample_<timestamp>.csv

Columns include:
    sample, dimension, method, y_label, target_gene, gRNA_simplified,
    effect_label, sil_score_train, sil_score_test
"""

# --- Import necessary libraries ---
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import silhouette_score
from optimisation import *

# --- Load and preprocess dataset ---
logcounts = pd.read_csv("/CRISPRi_whole_upr_logcounts.csv")  # Gene expression data
meta_data = pd.read_csv("/meta_CRISPRi_whole.csv")  # Metadata (gRNA, target gene, etc.)
data = preprocess_data(logcounts, meta_data)  # Merge and clean data for modeling

# --- Load pre-optimized configurations ---
import pickle
with open("Training Config/results_history_configs_CRISPRi_whole.pkl", "rb") as f:
    results_history_configs = pickle.load(f)

# --- Training hyperparameters ---
EPOCHS = 300
BATCH_SIZE = 256

# DataFrame to store results
effect_results_by_dim = pd.DataFrame(
    columns=["sample", "dimension", "method", "y_label", "target_gene",
             "gRNA_simplified", "effect_label", "sil_score_test", "sil_score_train"]
)

# List of latent dimensions to train over
latent_dim = [2, 3, 4, 5, 8, 12, 18, 27, 41, 62, 72, 92]

# Number of random samples to run
total_samples = 30

# --- Begin training loop ---
for sample in range(total_samples):
    # Split dataset into training and testing subsets
    X_train, X_test, target_gene_train, target_gene_test, gRNA_train, gRNA_test = (
        get_train_test_grna_sample(data)
    )

    # Iterate through label types (gRNA or target_gene)
    for y_label in ["gRNA", "target_gene"]:
        le = LabelEncoder()
        y_train = le.fit_transform(target_gene_train if y_label == "target_gene" else gRNA_train)
        y_test = le.transform(target_gene_test if y_label == "target_gene" else gRNA_test)

        num_classes = len(np.unique(y_train))
        best_models_euclidean = {}
        silhouette_score_history_euclidean = []

        # Train across all latent dimensions
        for dim in latent_dim:
            config = results_history_configs[y_label]["euclidean"]["best_models"][dim]["config"]

            print(f"Training: sample={sample}, latent_dim={dim}")

            # Build supervised autoencoder
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

            # Apply early stopping to prevent overfitting
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            # Train model
            history = autoencoder.fit(
                X_train,
                {"reconstruction": X_train, "classification": y_train},
                validation_split=0.2,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=0,
                callbacks=[early_stop]
            )

            # Encode train/test sets into latent space
            X_test_latent = encoder.predict(X_test)
            X_train_latent = encoder.predict(X_train)

            # --- Compute Euclidean distances and threshold ---
            distance_euclidean = compute_distance(X_train_latent, gRNA_train, method="euclidean")
            threshold_euclidean = compute_threshold_and_plot_hist(distance_euclidean, method="euclidean")

            # Label gRNAs as having an "effect" or "no effect"
            distance_euclidean["effect"] = np.where(distance_euclidean > threshold_euclidean, "effect", "no effect")
            effect_dict_euclidean = dict(zip(distance_euclidean.index, distance_euclidean["effect"]))
            true_labels_effect_test_euclidean = np.array([effect_dict_euclidean.get(g) for g in gRNA_test])
            true_labels_effect_train_euclidean = np.array([effect_dict_euclidean.get(g) for g in gRNA_train])

            # Compute Silhouette Scores for evaluation
            sil_score_test_euclidean = silhouette_score(X_test_latent, true_labels_effect_test_euclidean, metric='euclidean')
            sil_score_train_euclidean = silhouette_score(X_train_latent, true_labels_effect_train_euclidean, metric='euclidean')
            print("Test set Silhouette score:", sil_score_test_euclidean)
            print("Train set Silhouette score:", sil_score_train_euclidean)

            # Map effect labels back to dataset
            data_copy = data.copy()
            data_copy["effect"] = data_copy["gRNA_simplified"].map(effect_dict_euclidean)

            # Prepare result DataFrame for current configuration
            df = data_copy[["target_gene", "gRNA_simplified", "effect"]].drop_duplicates()
            df["effect_label"] = (df["effect"] == "effect").astype(int)

            # Add metadata
            df["sample"] = sample
            df["dimension"] = dim
            df["method"] = "euclidean"
            df["y_label"] = y_label
            df["sil_score_train"] = sil_score_train_euclidean
            df["sil_score_test"] = sil_score_test_euclidean

            # Append results
            effect_results_by_dim = pd.concat([effect_results_by_dim, df], ignore_index=True)

# --- Save results to timestamped CSV file ---
import os
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

os.makedirs("CRISPRi_whole_samples", exist_ok=True)
effect_results_by_dim.to_csv(f"CRISPRi_whole_samples/CRISPRi_whole_sample_{timestamp}.csv", index=False)