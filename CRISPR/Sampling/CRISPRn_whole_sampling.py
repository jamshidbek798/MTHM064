from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import silhouette_score

from optimisation import *


# Load expression data (log-transformed counts)
logcounts = pd.read_csv("Data/Dataset 2 — Perturb-seq/CRISPRn_whole_upr_logcounts.csv")

# Load sample metadata (gRNA, target gene, etc.)
meta_data = pd.read_csv("Data/Dataset 2 — Perturb-seq/meta_CRISPRn_whole.csv")

# Merge and clean data for modeling
data = preprocess_data(logcounts, meta_data)



import pickle

with open("Training Config/results_history_configs_CRISPRn_whole.pkl", "rb") as f:
    results_history_configs = pickle.load(f)


EPOCHS = 300
BATCH_SIZE = 256

effect_results_by_dim = pd.DataFrame(
    columns=["sample", "dimension", "method", "y_label", "target_gene", "gRNA_simplified", "effect_label", "sil_score_test", "sil_score_train"]
)
latent_dim = [2, 3, 4, 5, 8, 12, 18, 27, 41, 62, 72, 92]

total_samples = 1


for sample in range(total_samples):
    X_train, X_test, target_gene_train, target_gene_test, gRNA_train, gRNA_test = (
        get_train_test_grna_sample(data))

    for y_label in ["gRNA", "target_gene"]:


        le = LabelEncoder()
        # Fit on training set only (important to avoid data leakage)
        y_train = le.fit_transform(target_gene_train if y_label == "target_gene" else gRNA_train)
        # Transform test set using the same encoder
        y_test = le.transform(target_gene_test if y_label == "target_gene" else gRNA_test)

        num_classes = len(np.unique(y_train))
        best_models_euclidean = {}
        silhouette_score_history_euclidean = []



        for dim in latent_dim:
            config = results_history_configs[y_label]["euclidean"]['best_models'][dim]['config']


            print(f"Training: sample={sample}, latent_dim={dim}")

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

            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            history = autoencoder.fit(
                X_train,
                {"reconstruction": X_train, "classification": y_train},
                validation_split=0.2,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=0,
                callbacks=[early_stop]
            )

            # Encode train/test data
            X_test_latent = encoder.predict(X_test)
            X_train_latent = encoder.predict(X_train)

            # ---> EUCLIDEAN DISTANCE <---
            distance_euclidean = compute_distance(X_train_latent, gRNA_train, method="euclidean")
            threshold_euclidean = compute_threshold_and_plot_hist(distance_euclidean, method="euclidean")

            # Label effects
            distance_euclidean["effect"] = np.where(distance_euclidean > threshold_euclidean, "effect", "no effect")
            effect_dict_euclidean = dict(zip(distance_euclidean.index, distance_euclidean["effect"]))
            true_labels_effect_test_euclidean = np.array([effect_dict_euclidean.get(g) for g in gRNA_test])
            true_labels_effect_train_euclidean = np.array([effect_dict_euclidean.get(g) for g in gRNA_train])

            # Evaluate with Silhouette scores
            sil_score_test_euclidean = silhouette_score(X_test_latent, true_labels_effect_test_euclidean, metric='euclidean')
            sil_score_train_euclidean = silhouette_score(X_train_latent, true_labels_effect_train_euclidean, metric='euclidean')
            print("Test set Silhouette score:", sil_score_test_euclidean)
            print("Train set Silhouette score:", sil_score_train_euclidean)

            # results = distance.copy()
            # results["effect"] = np.where(results > threshold, "effect", "no effect")
            # effect_dict = dict(zip(results.index, results["effect"]))
            data_copy = data.copy()
            data_copy["effect"] = data_copy["gRNA_simplified"].map(effect_dict_euclidean)

            df = data_copy[["target_gene", "gRNA_simplified", "effect"]].drop_duplicates()
            df["effect_label"] = (df["effect"] == "effect").astype(int)

            # Add metadata columns
            df["sample"] = sample
            df["dimension"] = dim
            df["method"] = "euclidean"
            df["y_label"] = y_label
            df["sil_score_train"] = sil_score_train_euclidean
            df["sil_score_test"] = sil_score_test_euclidean

            effect_results_by_dim = pd.concat([effect_results_by_dim, df], ignore_index=True)



import os
import time

os.makedirs("CRISPRn_whole_samples", exist_ok=True)

effect_results_by_dim.to_csv(f"CRISPRn_whole_samples/CRISPRn_whole_sample_{int(time.time())}.csv", index=False)



