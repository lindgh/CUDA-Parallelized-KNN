import pandas as pd
import re as regex

digital_music = pd.read_csv("StratifiedSample.csv")

#SOME CLEANING
def clean_text(text):
  text = text.replace("<br />", " ") #remove line break tags
  text = regex.sub(r'[^\w\s]', " ", text) #removes every punctuation from the string
  return text.lower().strip() #converts all letters to lowercase


digital_music['clean_text'] = digital_music['text'].apply(clean_text)
digital_music['clean_title'] = digital_music['title'].apply(clean_text)

#COMBINE TITLE AND TEXT
digital_music['combine_title_text'] = digital_music['clean_title'] + " " + digital_music['clean_text']

#make validation set
rating_groups = digital_music.groupby('rating')
KNN_validation = rating_groups.tail(40) #last N entries of each group turns into validation
KNN_training = digital_music.drop(KNN_validation.index) #DROP VALIDATION SET FROM TRAINING

#MAKE SURE THAT EVERYTHING CAME OUT ALRIGHT
print(f"train size: {KNN_training.shape}")
print(f"validation size: {KNN_validation.shape}")
