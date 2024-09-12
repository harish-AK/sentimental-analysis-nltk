# sentimental-analysis-nltk

## Sentimental analysis on tweets based on crypto coins - [here](https://github.com/harish-AK/sentimental-analysis-nltk/blob/main/sentimental%20analysis.ipynb)
### 1. Libraries Used:
Data Handling and Visualization: pandas, numpy, matplotlib, seaborn
Text Preprocessing and Sentiment Analysis: TextBlob, tweepy, nltk, re
Modeling and Machine Learning: MultinomialNB, LogisticRegression, SVM, RandomForestClassifier, KNeighborsClassifier
Vectorization Techniques: CountVectorizer, TfidfVectorizer
Utility Libraries: joblib, scikit-learn for model evaluation (classification_report, confusion_matrix), wordcloud

### 2. Preprocessing Techniques:
Stopwords Removal, Lemmatization, and Stemming: nltk's WordNetLemmatizer, PorterStemmer, and stopwords are used for preprocessing the text data.
Text Tokenization and Vectorization: Techniques like CountVectorizer and TfidfVectorizer are applied to convert text into numerical features.

### 3. Data Splitting and Standardization:
Train-Test Split: The dataset is split into training and testing sets using train_test_split.
Standardization: Data is standardized using StandardScaler.

### 4. Machine Learning Models:
Several models are implemented for sentiment classification, including:
Naive Bayes (MultinomialNB)
Logistic Regression
SVM (Support Vector Machine)
K-Nearest Neighbors (KNN)
Random Forest Classifier

### 5. Evaluation Metrics:
Models are evaluated using metrics like accuracy, confusion matrix, and classification report (precision, recall, f1-score).

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Sentimental analysis on tweets based on tweets about 2019 Indian election - [here](https://github.com/harish-AK/sentimental-analysis-nltk/blob/main/modi%20vs%20rahul.ipynb)
### Data Cleaning and Preprocessing:

The notebook likely handles tweet data cleaning by removing unnecessary elements (e.g., URLs, special characters), typical in text-based machine learning tasks.
Sentiment Analysis:

Sentiment analysis is conducted using libraries like TextBlob, with results categorized into positive, negative, and neutral scores. This is essential for understanding the public's emotional reaction to each candidate's tweets.
Uncommon Aspect: It seems like a detailed comparison of sentiment for both Narendra Modi and Rahul Gandhi is performed using a merged dataset (dfNR), offering a combined view.
### Visualization:

Several visualizations, including bar plots, are generated using matplotlib to compare the overall sentiment polarity (positive, negative, neutral) for both Modi and Gandhi's tweets.
There is also visualization of the sentiment score distribution across all tweets, highlighting key differences between the two candidates.
### Topic Modeling:

The notebook likely includes topic modeling, a technique used to uncover hidden topics or themes within the tweet corpus. This is a more advanced, less common aspect compared to basic sentiment analysis.
### Emotion Scoring:
Beyond sentiment analysis, emotion scores (possibly through tools like NRC or VADER) are computed to provide a more granular view of emotions within the tweets.
