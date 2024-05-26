# Text Similarity and Key Phrase Extraction

This repository contains a comprehensive implementation for text similarity calculations, summarization, and key phrase extraction using various techniques and libraries.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Text Similarity](#text-similarity)
  - [Summarization](#summarization)
  - [Key Phrase Extraction](#key-phrase-extraction)
- [Methods](#methods)
  - [Text Similarity Methods](#text-similarity-methods)
  - [Summarization Methods](#summarization-methods)
  - [Key Phrase Extraction Methods](#key-phrase-extraction-methods)
- [Contributing](#contributing)


## Installation

To use the provided scripts, install the required packages:

```bash
pip install -r requirements.txt
```

# Requirements

The `requirements.txt` file should include:

```plaintext
scikit-learn
nltk
spacy
pke
rake-nltk
yake
networkx
transformers
tensorflow-hub
numpy
pandas
sentence-transformers
sumy
```

Additionally, download the necessary NLTK resources:

```
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
```
## Usage

### Text Similarity

Calculate text similarity using cosine similarity and Jaccard similarity. Refer to the script for implementation details.

### Summarization

Summarize text using various methods such as BERT, LSA, TextRank, TF-IDF, LexRank, Edmundson, and T5. Refer to the script for implementation details.

### Key Phrase Extraction

Extract key phrases using methods like TextRank, RAKE, YAKE, PatternRank, TopicRank, SingleRank, PositionRank, KPMiner, MultipartiteRank, and PromptRank. Refer to the script for implementation details.

## Methods

### Text Similarity Methods

- **Cosine Similarity**: Uses the cosine of the angle between two vectors in a multi-dimensional space.
- **Jaccard Similarity**: Measures similarity between finite sample sets, and is defined as the size of the intersection divided by the size of the union of the sample sets.

### Summarization Methods

- **BERT Summarization**: Uses the BERT model to generate summaries.
- **LSA Summarization**: Latent Semantic Analysis-based summarization.
- **TextRank Summarization**: Graph-based ranking algorithm for NLP.
- **TF-IDF Summarization**: Uses Term Frequency-Inverse Document Frequency for summarization.
- **LexRank Summarization**: A graph-based method using the concept of eigenvector centrality.
- **Edmundson Summarization**: Classic method based on features like term frequency.
- **T5 Summarization**: Uses the T5 model for text-to-text transfer transformations.

### Key Phrase Extraction Methods

- **TextRank**: Uses a graph-based ranking algorithm to extract key phrases.
- **RAKE**: Rapid Automatic Keyword Extraction algorithm.
- **YAKE**: Unsupervised key phrase extraction method.
- **PatternRank**: Extracts key phrases based on patterns.
- **TopicRank**: Clustering of similar key phrase candidates.
- **SingleRank**: Similar to TextRank but uses a single graph for ranking.
- **PositionRank**: Incorporates the position of words in the text for ranking.
- **KPMiner**: Extracts key phrases using statistical methods.
- **MultipartiteRank**: Graph-based ranking algorithm for key phrase extraction.
- **PromptRank**: Uses prompts to rank key phrases.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.
