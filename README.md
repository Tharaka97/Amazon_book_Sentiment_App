# Amazon Book Review Sentiment Analyzer

## ISY503 Intelligent Systems – Assessment 3

This project implements a Natural Language Processing (NLP) sentiment-analysis system for Amazon book reviews. A Convolutional Neural Network (CNN) classifies a review as either **Positive** or **Negative**.


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



