FAKE_REVIEW_SAMPLE_GENERATOR.py

i started from here to generate ramdom data and understand how the data could change final result

then GET_REVIEW_SCRAPER.py its a real working script with tor proxy for get reviews.csv file for get review,score,company as data from https://www.sitejabber.com

SCORE_AND_REVIEW_example.py is how the library should be implemented in the final code

# Competitor Finder Script

This Python script analyzes text content from websites and identifies the top N competitors based on content similarity using TF-IDF and cosine similarity. It is ideal for market research, SEO analysis, and competitive intelligence.

---

## Features

* **Data Loading**: Reads training data from a CSV file containing `content` and `website` columns.
* **Text Preprocessing**: Uses Scikit-learn’s `TfidfVectorizer` to convert text into TF-IDF features, with configurable parameters for stop words, document frequency thresholds, and text normalization.
* **Similarity Computation**: Computes cosine similarity between a new sample text and all existing documents to rank competitors.
* **Results Filtering**: Excludes the input website itself from the competitor list.
* **Persistence**: Saves the trained TF-IDF vectorizer to disk with `joblib` for later reuse.

---

## Prerequisites

* Python 3.8 or higher
* pip

Install required packages:

```bash
pip install pandas numpy scikit-learn joblib
```

---

## File Structure

```
competitor-finder/
├── training_data/
│   └── training_data.csv           # CSV with columns: 'content', 'website'
├── tfidf_vectorizer.joblib         # Saved vectorizer after first run
├── USE_COMPETITOR_FINDER_LIB.py    # Main script
└── README.md                       # This documentation
```

---

## Usage

1. **Prepare your training data**

   * Place a CSV file in `training_data/training_data.csv` with at least two columns:

     * `content`: Textual content of each webpage or document.
     * `website`: Unique identifier (e.g., URL or domain name).

2. **Run the script**

   ```bash
   python USE_COMPETITOR_FINDER_LIB.py
   ```

   By default, the script will:

   * Load `training_data/training_data.csv`.
   * Preprocess and vectorize all `content` entries.
   * Save the vectorizer to `tfidf_vectorizer.joblib`.
   * Compute similarity of a hardcoded sample input (`"buy a new domain"`).
   * Print the top 10 competitor websites and their similarity scores.

3. **Customize the input**

   * Edit the `sample_input` variable in the `__main__` section of `competitor_finder.py` to test different queries.
   * Optionally, modify the `training_data_path` or the number of competitors (`n`) in the call to `find_competitors()`.

---

## Functions Overview

### `load_data(file_path)`

Loads training data from a CSV file into a Pandas DataFrame.

**Arguments:**

* `file_path` (str): Path to the CSV file.

**Returns:**

* `DataFrame` containing the raw data.

---

### `preprocess_data(df)`

Extracts the `content` and `website` columns from the DataFrame for vectorization and identification.

**Arguments:**

* `df` (DataFrame): Raw data.

**Returns:**

* `X` (Series): Text content.
* `websites` (Series): Website identifiers.

---

### `find_competitors(model, vectorizer, sample_text, input_website, df, n=10)`

Computes cosine similarity between a sample text and the vectorized corpus, returning the top N competitors.

**Arguments:**

* `model` (sparse matrix): TF-IDF matrix of the corpus.
* `vectorizer` (TfidfVectorizer): Fitted vectorizer.
* `sample_text` (str): New text to compare.
* `input_website` (str): Identifier to exclude from results.
* `df` (DataFrame): Original dataset.
* `n` (int): Number of competitors to return (default: 10).

**Returns:**

* List of tuples: `(website, similarity_score)` sorted by descending similarity.

---

### `save_vectorizer(vectorizer, filename)`

Saves the trained TF-IDF vectorizer to a file using `joblib`.

**Arguments:**

* `vectorizer` (TfidfVectorizer): Fitted vectorizer.
* `filename` (str): Destination filepath.

---

## Extending and Integration

* **Batch queries**: Wrap `find_competitors()` in a loop or API endpoint to process multiple inputs.
* **CLI**: Enhance the script with `argparse` to pass file paths, sample text, and N via command-line flags.
* **Web Service**: Deploy as a REST API using FastAPI or Flask to serve competitor recommendations on demand.
* **Persistence**: Load the saved vectorizer (`tfidf_vectorizer.joblib`) to avoid retraining on each run.

---

## License

This script is released under the MIT License. Feel free to use, modify, and distribute.

---

*Last updated: July 28, 2025*
