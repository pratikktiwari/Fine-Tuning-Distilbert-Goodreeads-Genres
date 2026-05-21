# Fine-Tuning-Distilbert-Goodreeads-Genres

## Description

This project fine-tunes a [DistilBERT](https://huggingface.co/distilbert-base-cased) model to classify Goodreads book reviews by genre. The notebook downloads review data from the [UCSD Book Graph](https://mengtingwan.github.io/data/goodreads.html), trains a baseline logistic regression model for comparison, then fine-tunes DistilBERT using the HuggingFace `transformers` library. Experiment tracking is done with Weights & Biases, and the final model is pushed to the Hugging Face Hub.

**Genres classified:** Poetry, Children, Comics & Graphic, Fantasy & Paranormal, History & Biography, Mystery/Thriller/Crime, Romance, Young Adult.

## Setup

### Prerequisites

- Python 3.10+
- A GPU is recommended (the notebook was developed on Kaggle with an NVIDIA Tesla T4)

### Install Dependencies

```bash
pip install transformers accelerate wandb scikit-learn huggingface_hub torch matplotlib seaborn pandas
```

### Environment Variables / Secrets

The notebook expects the following secrets (configured via Kaggle Secrets or environment variables):

- `WANDB_API_KEY` [Weights & Biases](https://wandb.ai/) API key
- `HF_TOKEN` [Hugging Face](https://huggingface.co/settings/tokens) access token

If running locally, export them before launching Jupyter:

```bash
export WANDB_API_KEY="wandb-key"
export HF_TOKEN="hf-token"
```

### Run the Notebook

```bash
jupyter notebook mlops-fine-tuning-distilbert.ipynb
```

Run all cells sequentially. The notebook will:

1. Download and sample Goodreads reviews for each genre.
2. Train a TF-IDF + Logistic Regression baseline.
3. Tokenize the data with the DistilBERT tokenizer.
4. Fine-tune `distilbert-base-cased` for sequence classification.
5. Evaluate the model and log metrics to W&B.
6. Push the trained model to the Hugging Face Hub.

## Results

| Metric    | Score   |
| --------- | ------- |
| Accuracy  | 0.56375 |
| F1 Score  | 0.56289 |
| Eval Loss | 2.41293 |

## Links

- Kaggle Notebook: https://www.kaggle.com/code/pratikktiwari/mlops-fine-tuning-distilbert
- Hugging Face model: https://huggingface.co/pratikktiwari/fine-tuned-distilbert-goodreads-genres
- W&B Dashboard: https://wandb.ai/pratikktiwari-na/MLOps-Fine-Tuning-Distilbert-Goodreeads-Genres
- W&B report: https://api.wandb.ai/links/pratikktiwari-na/95t69xj4
