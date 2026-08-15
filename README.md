# Amazon Book Review Sentiment Analyzer

## ISY503 Intelligent Systems – Assessment 3

This project implements a Natural Language Processing (NLP) sentiment-analysis system for Amazon book reviews. A Convolutional Neural Network (CNN) classifies a review as either **Positive** or **Negative**.

## Final model results

- Test accuracy: **75.51%**
- Negative class: precision **0.75**, recall **0.76**, F1-score **0.75**
- Positive class: precision **0.76**, recall **0.75**, F1-score **0.76**
- Test set size: **294 reviews**

Confusion matrix results:

| Actual class | Predicted Negative | Predicted Positive |
|---|---:|---:|
| Negative | 110 | 35 |
| Positive | 37 | 112 |

## Project workflow

1. Load positive and negative Amazon book reviews.
2. Explore and randomise the dataset.
3. Clean review text.
4. Remove very short/invalid reviews.
5. Encode sentiment labels (`0 = Negative`, `1 = Positive`).
6. Tokenise review text.
7. Pad/truncate sequences to 250 tokens.
8. Split the data into training, validation and test sets.
9. Compare several neural-network architectures.
10. Select a 1D CNN based on validation performance.
11. Evaluate the final model on the untouched test set.
12. Deploy the trained model through a Streamlit web interface.

## Model architecture

The final model uses:

- Embedding layer
- 1D Convolution (`Conv1D`)
- Global Max Pooling
- Dropout
- Dense layer with L2 regularisation
- Sigmoid output layer

## Required files

Place the following files in the repository root:

```text
amazon-book-sentiment-analyzer/
├── app.py
├── best_cnn_book_model.keras
├── tokenizer.pkl
├── requirements.txt
├── README.md
├── .gitignore
└── notebooks/
    └── ISY503_Book_Review_Sentiment_Analysis.ipynb
```

`best_cnn_book_model.keras` and `tokenizer.pkl` must be the exact files produced during model training.

## Run locally

Create a virtual environment if desired, then install the dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

Streamlit will display a local URL. Open it in your browser.

## Example reviews

Positive example:

```text
This book was fantastic. The story was engaging and the characters were beautifully written.
```

Negative example:

```text
This book was extremely boring and poorly written. I regret buying it.
```

Mixed example:

```text
The story was interesting but the ending was very disappointing.
```


## Limitations and ethical considerations

The model performs binary classification, so it may struggle with mixed sentiment, sarcasm, context-dependent language, spelling variations and reviews that differ substantially from the training data. Dataset labels may also contain subjectivity or errors. Predictions should therefore be treated as model estimates rather than objective judgments.

## Team contribution

Add the team member names, student IDs and contribution information required for the assessment here or in the individual contribution report.

## Dataset

The project uses the Amazon Multi-Domain Sentiment Dataset specified in the ISY503 assessment brief, using the **Books** domain.

## Academic integrity

Clearly identify your team's own code and any external material used. Add citations/references required by your subject's academic-integrity and APA requirements.
