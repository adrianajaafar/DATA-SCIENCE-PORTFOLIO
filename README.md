# Data-Science-Portfolio
# Project 1: Reddit Comment Analysis and Emotion Detection

## Overview

This project encompasses two primary tasks:
1. **Reddit Comment Scraper**: Scraping comments from specific Reddit posts and storing them in an Excel file.
2. **Emotion Detection and Sentiment Analysis**: Analyzing the emotions and sentiments expressed in the scraped Reddit comments using NLP and neural networks.

## Getting Started

### Prerequisites

- Python 3.x
- PRAW
- Pandas
- Matplotlib
- Seaborn
- NLTK
- Transformers
- TextBlob
- Scikit-learn
- TensorFlow
- Joblib

### Installation

### :information_source: Where I can get the reddit parameters?

- Parameters indicated with `<...>` on the previous script
- Official [Reddit guide](https://github.com/reddit-archive/reddit/wiki/OAuth2)
- TLDR: read this [stack overflow](https://stackoverflow.com/a/42304034)

| Parameter name | Description | How get it| Example of the value |
| --- | --- | --- | --- |
| `reddit_id` | The Client ID generated from the apps page | [Official guide](https://github.com/reddit-archive/reddit/wiki/OAuth2#authorization-implicit-grant-flow) | 50oK80pF8ac3Cn |
| `reddit_secret` | The secret generated from the apps page | Copy the value as showed [here](https://github.com/reddit-archive/reddit/wiki/OAuth2#getting-started) | 9KEUOE7pi8dsjs9507asdeurowGCcgi|
| `reddit_username` | The reddit account name| The name you use for log in | pistoSniffer |
