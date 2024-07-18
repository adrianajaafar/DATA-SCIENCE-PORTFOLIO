# Data-Science-Portfolio
# Project 1: Reddit Comment Analysis and Emotion Detection 

## Overview
- Aims to analyze sentiment and emotion in Reddit comments from three subreddits related to AI and job security.
- Scraping comments from specific posts
- Performing comprehensive emotion detection and sentiment analysis using natural language processing (NLP) and neural networks.
- Gain insights into public perceptions and concerns about the impact of AI on job security.

## Objective
- To develop a model that can accurately predict the sentiment and emotion of Reddit comments related to AI and job security.
- To understand the overall public sentiment and specific emotional responses to the perceived impact of AI on employment.

## Getting Started

### This project encompasses two primary tasks:
1. **Reddit Comment Scraper**: Scraping comments from specific Reddit posts and storing them in an Excel file.
2. **Emotion Detection and Sentiment Analysis**: Analyzing the emotions and sentiments expressed in the scraped Reddit comments using NLP and neural networks.

## Prerequisites

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

## Installation
Install the required Python packages using pip:
```bash
pip install praw pandas matplotlib seaborn nltk transformers textblob scikit-learn tensorflow joblib
```

## Data Collection
### Reddit Comment Scrapper
Using the PRAW (Python Reddit API Wrapper) library, scraped comments from three specific posts related to AI and job security:

1. Are you guys worried that AI will take your job? https://www.reddit.com/r/WFH/comments/17i2fhi/are_you_guys_worried_that_ai_will_take_your_job/
2. Is anyone concerned about the impact AI might have on jobs? https://www.reddit.com/r/UKJobs/comments/19ebmm1/is_anyone_concerned_about_the_impact_ai_might/
3. Is AI causing a massive wave of unemployment now? https://www.reddit.com/r/OpenAI/comments/1afv431/is_ai_causing_a_massive_wave_of_unemployment_now/
   
All comments were compiled into a combined dataset for further analysis.

### :information_source: Where I can get the reddit parameters?

- Parameters indicated with `<...>` on the previous script
- Official [Reddit guide](https://github.com/reddit-archive/reddit/wiki/OAuth2)
- TLDR: read this [stack overflow](https://stackoverflow.com/a/42304034)

| Parameter name | Description | How get it| Example of the value |
| --- | --- | --- | --- |
| `reddit_id` | The Client ID generated from the apps page | [Official guide](https://github.com/reddit-archive/reddit/wiki/OAuth2#authorization-implicit-grant-flow) | fYowxtbWbDueM9q_qXLP_g |
| `reddit_secret` | The secret generated from the apps page | Copy the value as showed [here](https://github.com/reddit-archive/reddit/wiki/OAuth2#getting-started) | IfEpaUgcf1DUDfYyb0q0id4C_nSRIg|
| `reddit_username` | The reddit account name| The name you use for log in | No-Delivery-80883 |

## Data Processing and Analysis
#### Preprocessing
Text preprocessing steps included:

- Converting text to lowercase
- Removing punctuation
- Tokenizing text
- Removing stopwords
- Lemmatizing words

These steps ensured that the text data was clean and standardized for further analysis.

### Emotion Detection
- Used **"j-hartmann/emotion-english-distilroberta-base"** model from **Hugging Face** for emotion detection.
- Each comment was analyzed to detect one of seven emotions: **anger, disgust, fear, joy, neutral, sadness,** and **surprise**.

### Sentiment Analysis
**TextBlob** was used to calculate the polarity and subjectivity of each comment:

**- Polarity:** Ranges from **-1 (negative) to 1 (positive)**, indicating the overall sentiment.

**- Subjectivity:** Ranges from **0 (objective) to 1 (subjective)**, indicating how opinionated the text is.

### Model Training
A neural network model was built and trained to predict the emotion of comments. Key steps included:

- Encoding the target variable (emotion labels)
- Splitting the data into training and testing sets
- Vectorizing the text data using TF-IDF
- Building and training a neural network with the vectorized text data
- Evaluating the model's performance
  
## Results

### Sentiment Analysis

**- Average Polarity:** The overall sentiment of the comments was analyzed, revealing a slightly negative tendency, indicating concerns about AI's impact on job security.

**- Average Subjectivity:** The analysis indicated that the comments were moderately subjective, reflecting personal opinions and experiences.

### Emotion Analysis

**- Emotion Distribution:** The most common emotions detected were **fear and sadness**, highlighting significant concern and anxiety about AI's impact on job security.

**- Emotion Percentages:** A detailed breakdown showed the prevalence of each emotion, with fear and sadness being the most dominant emotions.

### Model Performance

**- Accuracy:** The neural network model achieved an **accuracy of around 85%** on the test data.

**- Confusion Matrix:** The matrix showed the model's performance in **accurately** classifying each emotion, with higher accuracy for more common emotions like fear and sadness.

## Conclusion

- A significant portion of Reddit users express fear and sadness regarding AI and job security, reflecting concerns about potential job displacement due to technological advancements.
- Sentiments observed in the comments generally lean towards negativity, indicating a widespread apprehension about AI's impact on employment stability.
- Addressing these fears through informed public discourse and policy-making is crucial to mitigate anxieties and promote societal acceptance of AI technologies.
- The trained neural network model accurately predicts emotions in new comments, enhancing our ability to monitor and understand public sentiment on AI and job security.
- This project illustrates the efficacy of integrating web scraping, NLP, and machine learning to derive actionable insights from social media data.
- These insights can inform decision-making and strategic planning, aiding in navigating the societal implications of technological progress effectively.

## Future Work
Future improvements to this project could include:

- Expanding the dataset to include more subreddits and posts for a broader analysis.
- Enhancing the emotion detection model with additional training data.
- Implementing more advanced sentiment analysis techniques.
- Creating a real-time dashboard to visualize sentiment and emotion trends over time.

By continuously refining these methods, we can gain deeper insights into public sentiment and emotional responses to AI and its impact on job security.

