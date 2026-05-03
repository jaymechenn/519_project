python src/data_processing_eda.py
python src/prepare_model_inputs.py
python src/train_baseline_models.py

data_processing_eda.py creates cleaned train/validation/test splits.
prepare_model_inputs.py creates TF-IDF matrices, advanced feature CSVs, and transformer-ready CSVs.
train_baseline_models.py trains the initial TF-IDF + logistic regression baseline.