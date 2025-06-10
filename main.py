import pandas as pd
import re as regex
from sklearn.feature_extraction.text import TfidfVectorizer

digital_music = pd.read_csv("StratifiedSample.csv")

#SOME CLEANING
def clean_text(text):
  text = text.replace("<br />", " ")
  text = regex.sub(r'[^\w\s]', " ", text) 
  return text.lower().strip() 


digital_music['clean_text'] = digital_music['text'].apply(clean_text)
digital_music['clean_title'] = digital_music['title'].apply(clean_text)

#COMBINE TITLE AND TEXT
digital_music['combine_title_text'] = digital_music['clean_title'] + " " + digital_music['clean_text']

#MAKE TRAINING AND VALIDATION SETS
rating_groups = digital_music.groupby('rating')
KNN_validation = rating_groups.tail(40) #40 validation from each rating
KNN_training = digital_music.drop(KNN_validation.index) #160 training from each rating


#DROP ALL THE USELESS STUFF 
#KNN_validation = 

#MAKE SURE THAT EVERYTHING CAME OUT ALRIGHT
print(f"train size: {KNN_training.shape}")
print(f"validation size: {KNN_validation.shape}")

#tfidf = term frequency * inverse document frequency
# tf = how often word appears in document
# idf = how rare word is across all documents
# stopwords remove common englihs words, max features means 3000 most frequent words
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)

#input features for training and testing 
x_train = vectorizer.fit_transform(KNN_training['combine_title_text']).toarray()
x_val = vectorizer.transform(KNN_validation['combine_title_text']).toarray()

#target predictions
#true label
y_train = KNN_training['rating'].values  
y_val = KNN_validation['rating'].values
print(f"tf-idf vectorized training set size: {x_train.shape}")
print(f"tf-idf vectorized validation set size: {x_val.shape}")

#ACTUAL KERNELS:

#CPU ONLY:


#NUMBA KERNEL:

