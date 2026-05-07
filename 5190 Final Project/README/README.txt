CIS 5190 Final Project: News Source Classification
===================================================

Project goal
------------
This project predicts whether a news headline came from NBC or FoxNews.
The model uses headline text as input. URLs are used only to create labels:

- NBC = 0
- FoxNews = 1

The final submitted model is a PyTorch-compatible checkpoint model saved at:

models/news_source_pytorch.pt

It uses hashed word and character TF-IDF-style features with a linear classifier.


Project structure
-----------------
Resources/csv/url_with_headlines.csv
    Original raw data with URL and headline columns.

Resources/processed/
    Cleaned train, validation, and test CSV splits.

src/data_processing_eda.py
    Cleans the raw data, creates labels from URLs, saves train/validation/test
    splits, and generates EDA plots.

src/baseline_model.py
    Trains the baseline TF-IDF + Logistic Regression model.

src/ensemble_model.py
    Trains a soft-voting ensemble of TF-IDF text classifiers.

src/train_pytorch_model.py
    Trains the final PyTorch-compatible checkpoint model.

src/model.py
    Defines the evaluator-facing NewsClassifier / Model class.

src/preprocess.py
    Defines prepare_data(csv_path), which returns headline inputs and labels.

src/eval_project_b.py
    Local evaluator for model.py + preprocess.py + optional .pt checkpoint.

models/
    Saved trained models and checkpoints.

outputs/
    Metrics files, confusion matrices, and EDA plots.


How to reproduce
----------------
From the "5190 Final Project" directory:

1. Clean data and generate train/validation/test splits:

    python src/data_processing_eda.py

2. Train the baseline model:

    python src/baseline_model.py

3. Train the soft-voting ensemble:

    python src/ensemble_model.py

4. Train the final PyTorch checkpoint model:

    python src/train_pytorch_model.py

5. Evaluate the final checkpoint model:

    python src/eval_project_b.py \
      --model src/model.py \
      --preprocess src/preprocess.py \
      --csv Resources/processed/test.csv \
      --weights models/news_source_pytorch.pt \
      --batch-size 64


Current results
---------------
Baseline TF-IDF + Logistic Regression:

- Validation accuracy: 0.7750
- Test accuracy: 0.7789

Soft-voting TF-IDF ensemble:

- Validation accuracy: 0.7908
- Test accuracy: 0.8026

Final PyTorch checkpoint model:

- Validation accuracy: 0.8276
- Test accuracy: 0.8434

The final checkpoint model is the best-performing model in the current project.


Final model summary
-------------------
The final model combines two feature views:

1. Word hashed TF-IDF-style features
   - word unigrams and bigrams
   - English stopword filtering
   - 131,072 hashed dimensions

2. Character hashed TF-IDF-style features
   - char_wb n-grams from length 2 to 5
   - 131,072 hashed dimensions

The two feature blocks are IDF-weighted, L2-normalized separately, concatenated,
and passed through a PyTorch linear classifier. The learned IDF values and
linear classifier weights are stored in models/news_source_pytorch.pt.


Important notes
---------------
- The URL is not used as a predictive feature.
- The current preprocess.py performs lightweight headline cleaning and label
  extraction for raw or processed CSV files.
- The final model is classical text classification wrapped in a PyTorch module
  so it works with the project evaluator's .pt checkpoint interface.
- The soft-voting ensemble is useful for comparison, but it is not the strongest
  final model.
