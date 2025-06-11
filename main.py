import pandas as pd
import re as regex
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from naive_knn import run_naive_knn
from naive_K_search import naive_bestKsearch


print("\n=============== CUDA PARALLELIZED K-NEAREST NEIGHBOR CLASSIFICATION ===============")

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
print("\n=============== CREATING TEST AND VALIDATION SETS ===============")

#MAKE SURE THAT EVERYTHING CAME OUT ALRIGHT
print(f"train size: {KNN_training.shape}")
print(f"validation size: {KNN_validation.shape}")

#tfidf = term frequency * inverse document frequency
# tf = how often word appears in document
# idf = how rare word is across all documents
# stopwords remove common englihs words, max features means 3000 most frequent words
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)

print("\n----- TF-IDF vectorizing -----")

#input features for training and testing 
x_train = vectorizer.fit_transform(KNN_training['combine_title_text']).toarray()
x_val = vectorizer.transform(KNN_validation['combine_title_text']).toarray()

#MAKE DENSE VECTORS
svd = TruncatedSVD(n_components=100, random_state=42)
X_train = svd.fit_transform(x_train)
X_val = svd.transform(x_val)

#target predictions
#true label
y_train = KNN_training['rating'].values  
y_val = KNN_validation['rating'].values
print(f"tf-idf vectorized training set size: {x_train.shape}")
print(f"tf-idf vectorized validation set size: {x_val.shape}")


print("\n=============== KNN CLASSIFICATION ===============")

run_naive_knn(X_train, y_train, X_val, y_val, k = 3)


print("\n=============== FINDING THE BEST K VALUE ===============")

naive_bestKsearch(X_train, y_train, X_val, y_val, 15)


