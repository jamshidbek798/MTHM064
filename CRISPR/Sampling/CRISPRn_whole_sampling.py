"""
This script trains Supervised Autoencoder (SAE) models for the CRISPRn_whole dataset in a cloud environment.

Description:
------------
The script automates the training and evaluation of supervised autoencoders for large-scale single-cell CRISPRn
perturbation data. It uses previously optimized hyperparameter configurations to train models across multiple
latent dimensions and identifies perturbation effects (effect vs no effect) using Euclidean distance in the latent space.

Workflow:
---------
1. Loads preprocessed CRISPRn_whole expression and metadata files.
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
    CRISPRn_whole_samples/CRISPRn_whole_sample_<timestamp>.csv

The file contains:
    sample, dimension, method, y_label, target_gene, gRNA_simplified,
    effect_label, sil_score_train, sil_score_test
"""

# --- Import necessary libraries ---
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import silhouette_score
from optimisation import *

# --- Load dataset ---
# Gene expression matrix (log-transformed counts)
logcounts = pd.read_csv("/CRISPRn_whole_upr_logcounts.csv")
# Sample metadata (gRNA, target gene, etc.)
meta_data = pd.read_csv("/meta_CRISPRn_whole.csv")

# Merge and clean data
data = preprocess_data(logcounts, meta_data)

# --- Load pre-optimized model configurations ---
import pickle
with open("Training Config/results_history_configs_CRISPRn_whole.pkl", "rb") as f:
    results_history_configs = pickle.load(f)

# --- Define training parameters ---
EPOCHS = 300
BATCH_SIZE = 256

# Initialize a DataFrame to store model results
effect_results_by_dim = pd.DataFrame(
    columns=["sample", "dimension", "method", "y_label", "target_gene", "gRNA_simplified",
             "effect_label", "sil_score_test", "sil_score_train"]
)

# Define latent dimensions to explore
latent_dim = [2, 3, 4, 5, 8, 12, 18, 27, 41, 62, 72, 92]

# Number of random training samples to run
total_samples = 30

# --- Model training loop ---
for sample in range(total_samples):
    # Split data into training and testing subsets
    X_train, X_test, target_gene_train, target_gene_test, gRNA_train, gRNA_test = (
        get_train_test_grna_sample(data)
    )

    # Iterate over different labeling levels
    for y_label in ["gRNA", "target_gene"]:

        # Encode labels for supervised training
        le = LabelEncoder()
        y_train = le.fit_transform(target_gene_train if y_label == "target_gene" else gRNA_train)
        y_test = le.transform(target_gene_test if y_label == "target_gene" else gRNA_test)
        num_classes = len(np.unique(y_train))

        best_models_euclidean = {}
        silhouette_score_history_euclidean = []

        # --- Iterate over each latent dimension ---
        for dim in latent_dim:
            # Load hyperparameter configuration for current dimension
            config = results_history_configs[y_label]["euclidean"]['best_models'][dim]['config']

            print(f"Training: sample={sample}, latent_dim={dim}")

            # Build and compile the supervised autoencoder
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

            # Define early stopping criteria
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            # Train the autoencoder
            history = autoencoder.fit(
                X_train,
                {"reconstruction": X_train, "classification": y_train},
                validation_split=0.2,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=0,
                callbacks=[early_stop]
            )

            # Encode both train and test data into latent space
            X_test_latent = encoder.predict(X_test)
            X_train_latent = encoder.predict(X_train)

            # --- Compute Euclidean distance between gRNA groups ---
            distance_euclidean = compute_distance(X_train_latent, gRNA_train, method="euclidean")
            threshold_euclidean = compute_threshold_and_plot_hist(distance_euclidean, method="euclidean")

            # Assign 'effect' or 'no effect' based on threshold
            distance_euclidean["effect"] = np.where(distance_euclidean > threshold_euclidean, "effect", "no effect")
            effect_dict_euclidean = dict(zip(distance_euclidean.index, distance_euclidean["effect"]))
            true_labels_effect_test_euclidean = np.array([effect_dict_euclidean.get(g) for g in gRNA_test])
            true_labels_effect_train_euclidean = np.array([effect_dict_euclidean.get(g) for g in gRNA_train])

            # --- Evaluate latent space using Silhouette Scores ---
            sil_score_test_euclidean = silhouette_score(X_test_latent, true_labels_effect_test_euclidean, metric='euclidean')
            sil_score_train_euclidean = silhouette_score(X_train_latent, true_labels_effect_train_euclidean, metric='euclidean')
            print("Test set Silhouette score:", sil_score_test_euclidean)
            print("Train set Silhouette score:", sil_score_train_euclidean)

            # Map gRNA effects back to dataset
            data_copy = data.copy()
            data_copy["effect"] = data_copy["gRNA_simplified"].map(effect_dict_euclidean)

            # Create result DataFrame for current configuration
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

# --- Save all results ---
import os
import time
os.makedirs("CRISPRn_whole_samples", exist_ok=True)
effect_results_by_dim.to_csv(f"CRISPRn_whole_samples/CRISPRn_whole_sample_{int(time.time())}.csv", index=False)