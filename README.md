# Data-Science-Portfolio
# PROJECT 1: Reddit Comment Analysis and Emotion Detection 

## OVERVIEW
- Aims to analyze sentiment and emotion in Reddit comments from three subreddits related to AI and job security.
- Scraping comments from specific posts
- Performing comprehensive emotion detection and sentiment analysis using natural language processing (NLP) and neural networks.
- Gain insights into public perceptions and concerns about the impact of AI on job security.

## OBJECTIVE
- To develop a model that can accurately predict the sentiment and emotion of Reddit comments related to AI and job security.
- To understand the overall public sentiment and specific emotional responses to the perceived impact of AI on employment.

## Getting Started

### This project encompasses two primary tasks:
1. **Reddit Comment Scraper**: Scraping comments from specific Reddit posts and storing them in an Excel file.
2. **Emotion Detection and Sentiment Analysis**: Analyzing the emotions and sentiments expressed in the scraped Reddit comments using NLP and neural networks.

## Prerequisite

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

## DATA COLLECTION
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

## DATA PROCESSING

#### Preprocessing
Text preprocessing steps included:

- Converting text to lowercase
- Removing punctuation
- Tokenizing text
- Removing stopwords
- Lemmatizing words

These steps ensured that the text data was clean and standardized for further analysis.

## ANALYSIS

### 1.   Emotion Detection

- Used **"j-hartmann/emotion-english-distilroberta-base"** model from **Hugging Face** for emotion detection.
- Each comment was analyzed to detect one of seven emotions: **anger, disgust, fear, joy, neutral, sadness,** and **surprise**.

### 2.   Sentiment Analysis

**TextBlob** was used to calculate the polarity and subjectivity of each comment:

**- Polarity:** Ranges from **-1 (negative) to 1 (positive)**, indicating the overall sentiment.

**- Subjectivity:** Ranges from **0 (objective) to 1 (subjective)**, indicating how opinionated the text is.

### 3.   Model Training

A neural network model was built and trained to predict the emotion of comments. Key steps included:

- Encoding the target variable (emotion labels)
- Splitting the data into training and testing sets
- Vectorizing the text data using TF-IDF
- Building and training a neural network with the vectorized text data
- Evaluating the model's performance
  
## RESULTS

### 1.   Sentiment Analysis

- **Average Polarity:** 0.082
The average polarity score suggests a slightly positive sentiment overall across the             analyzed comments.

- **Average Subjectivity:** 0.385
The average subjectivity score indicates that the comments tend to be more objective             rather than subjective in nature.

- **Conclusion:** Overall positive sentiment
Based on the average polarity score, the sentiment analysis concludes that the overall           sentiment expressed in the comments leans towards positivity.

### 2.   Emotion Analysis

**- Emotion Distribution:** The most common emotions detected were **fear and sadness**, highlighting significant concern and anxiety about AI's impact on job security.

**- Emotion Percentages:** A detailed breakdown showed the prevalence of each emotion, with fear and sadness being the most dominant emotions.

### 3.   Model Performance

**Accuracy:** 41.81%
The accuracy of the emotion detection model on the test data is 41.81%. This metric indicates the overall correctness of the model's predictions compared to the actual labels.

**Classification Report**
The classification report provides insights into the precision, recall, and F1-score for each emotion category:

- **Precision:** Indicates the proportion of correctly identified instances among the predicted instances for each emotion. For example, the precision for 'fear' is 0.57, suggesting that when the model predicts 'fear', it is correct 57% of the time.

- **Recall:** Indicates the proportion of correctly identified instances among the actual instances for each emotion. For instance, the recall for 'neutral' is 0.78, meaning the model correctly identifies 78% of all actual 'neutral' instances.

- **F1-score:** The weighted average of precision and recall. For instance, 'neutral' has an F1-score of 0.64, indicating a good balance between precision and recall for this emotion.


**Confusion Matrix:** The confusion matrix provides a more granular view of the model's performance by showing the number of correct and incorrect predictions for each emotion class. Here's a detailed breakdown:
```bash
[[ 0  0  0  3  6  3  2]  # anger
 [ 0  0  0  0  1  0  0]  # disgust
 [ 1  0  4  3  8  1  1]  # fear
 [ 1  0  1  3  8  3  7]  # joy
 [ 2  0  1  6 53  0  6]  # neutral
 [ 1  0  0  0  9  1  6]  # sadness
 [ 1  0  1  3 13  5 13]] # surprise
```
 
## CONCLUSION

The analysis revealed several key insights about the sentiment and emotion surrounding AI and job security based on Reddit comments. Here's a summary:

1. **Overall Sentiment:**

- The average polarity score of 0.082 suggests a slightly positive sentiment overall. Despite concerns about job security, users tend to express a mildly positive outlook towards AI.
  
2. **Emotion Distribution:**

- The most common emotion expressed is 'neutral' (39.66%), indicating a balanced or indifferent stance on the topic.
  
- 'Surprise' (15.93%) and 'sadness' (13.67%) are also significant, reflecting unexpected reactions and concerns about AI and job security.
  
- Emotions such as 'joy' (11.30%), 'fear' (9.60%), and 'anger' (9.49%) highlight the mixed feelings of users, ranging from positive aspects to anxiety and frustration.
  
- 'Disgust' is the least common emotion (0.34%).
  
3. **Model Performance:**

- The neural network model achieved an accuracy of 41.81%, with the best performance in predicting 'neutral' emotions.
  
- The classification report and confusion matrix indicate areas for improvement, particularly in distinguishing between similar emotions such as 'joy' and 'surprise'.
  
4. **Practical Application:**

- The trained model can predict emotions in new comments, providing a tool for monitoring public sentiment on AI and job security.
  
- This capability can inform decision-making, strategic planning, and public discourse by highlighting prevailing concerns and positive reactions.
  
5. **Actionable Insights:**

- The analysis shows a significant portion of Reddit users express fear and sadness about AI's impact on job security. Addressing these fears through public dialogue and policy-making is crucial.
  
- The combination of web scraping, NLP, and machine learning demonstrated in this project showcases the potential to extract valuable insights from social media data, aiding in understanding societal impacts of technological advancements.
  
In conclusion, this project highlights the power of advanced data analytics to uncover deep insights from social media discussions, which can be pivotal for stakeholders in making informed decisions regarding AI and its implications for the workforce.

## Recommendation
Future improvements to this project could include:

- Expanding the dataset to include more subreddits and posts for a broader analysis.

- Ensure a more balanced distribution of emotions to avoid bias towards more frequent emotions like 'neutral'.
  
-  Utilize advanced embedding techniques like BERT or GPT to capture the context of comments more accurately. These models can understand the nuances and complexities of human language better than traditional methods.\
 
- Implement cross-validation to ensure the model's performance is consistent across different subsets of the data, leading to more reliable results.
  
-  Implement a pipeline for regular updates to the model with new data, ensuring it stays relevant and accurate over time.

By continuously refining these methods, it significantly enhance the robustness, accuracy, and applicability of your data modeling and analysis, leading to more insightful and actionable results..

