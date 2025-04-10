import os
import re
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from joblib import dump, load

import constants
from feature_extraction import feature_extraction

# contains mapping such as "don't" => "do not"
appos = constants.appos
stopwords = constants.stopwords

scoring = ['accuracy', 'precision_macro', 'recall_macro']

# normalizing exaggerated words
def reduce_lengthening(text):
	pattern = re.compile(r"([a-zA-Z])\1{2,}")
	return pattern.sub(r"\1\1", text)

def preprocess(txt, nlp):
	txt = txt.lower() # converting text to lower case
	txt = reduce_lengthening(txt) # normalizing exaggerated words
	
	with nlp.disable_pipes('tagger', 'parser', 'ner'):
		doc = nlp(txt) # tokenizing the words
	
	tokens = [token.text for token in doc]
	
	# removing reviews with less than 3 tokens
	if len(tokens) <3:
		return np.NaN
	
	# normalizing words with apostrophe
	for i, token in enumerate(tokens):
		if token in appos:
			tokens[i] = appos[token]
			
	txt = ' '.join(tokens)
	txt = re.sub(r"[^a-zA-Z. \n]", " ", txt)
	txt = re.sub(r"([. \n])\1{1,}", r"\1", txt)
	txt = re.sub(r" ([.\n])", r"\1", txt)
	txt = re.sub(r" ?\n ?", ".", txt)
	txt = re.sub(r"([. \n])\1{1,}", r"\1", txt)
	
	return txt.strip()

def postprocess(x, nlp):
	# removing stop words
	with nlp.disable_pipes(['tagger', 'parser', 'ner', 'sentencizer']):
		doc = nlp(x)
	
	words = [token.text for token in doc if token.text not in stopwords]
	x = ' '.join(words)
	x = re.sub(r"[0-9\n.?:;,-]", " ", x)
	x = re.sub(r"[ ]{2,}", " ", x)
	
	return x

def construct_spacy_obj(df, nlp):
	with nlp.disable_pipes(['parser', 'ner', 'sentencizer']):
		# constructing spacy object for each review
		docs = list(nlp.pipe(df['reviewText']))
		df['spacyObj'] = pd.Series(docs, index=df['reviewText'].index)
	
	return df

def get_sigle_aspect_reviews(*dfs, features):
	#count reviews that talk about only one aspect
	total_count = 0
	reviews = []
	ratings = []

	# all_features = ['android', 'battery', 'camera', 'charger', 'charging', 'delivery', 'device', 'display', 'features', 'fingerprint', 'gaming', 'issue', 'mode', 'money', 'performance', 'phone', 'price', 'problem', 'product', 'screen']

	for df in dfs: 
		for i, review in df['spacyObj'].items():
			flag = True
			found = set()

			for token in review:
				if token.text in features:
					if len(found) <3:
						found.add(token.text)
					elif token.text not in found:
						flag = False
						break

			if flag:
				total_count += 1
				reviews.append(review.text)
				ratings.append(df['rating'][i])
	
	print(total_count)
	return pd.DataFrame({'reviewText': reviews, 'rating': ratings})

def giveRating(x):
	if x in [5,4]:
		return "Positive"
	elif x in [1,2,3]:
		return "Negative"

def get_model(nlp, ft_model):

	if os.path.isfile('models/model.joblib'):
		print("✅ Trained model found. Loading the model.")
		model = load('models/model.joblib')

	else:
		print("🚀 Trained models not found. Starting training process...\n")

		# Step 1: Load & preprocess training data
		print("📥 Loading training data...")
		train_data = pd.read_csv('csv_files/training.csv', header=None, names=['reviewText', 'rating'])
		train_data.dropna(inplace=True)

		print("🧹 Preprocessing text data...")
		train_data['reviewText'] = train_data['reviewText'].apply(lambda x: preprocess(x, nlp))
		train_data.dropna(inplace=True)
		print("✅ Preprocessing done. Total samples:", len(train_data))

		# Step 2: Create SpaCy object
		print("\n🧠 Creating SpaCy objects (tokenizing reviews)...")
		train_data = construct_spacy_obj(train_data, nlp)
		print("✅ SpaCy tokenization complete.")

		# Step 3: Feature extraction using FastText
		print("\n📊 Extracting features using FastText...")
		features = feature_extraction(train_data, ft_model, nlp)
		print("✅ Feature extraction complete. Total features:", len(features))

		# Step 4: Filter single-aspect reviews
		print("\n🔍 Filtering single-aspect reviews...")
		single_aspect_reviews = get_sigle_aspect_reviews(train_data, features=features)
		print("✅ Single-aspect review filtering complete. Total samples:", len(single_aspect_reviews))

		# Step 5: Final postprocessing
		print("\n🧽 Postprocessing text data (removing stopwords, etc.)...")
		single_aspect_reviews['reviewText'] = single_aspect_reviews['reviewText'].apply(lambda x: postprocess(x, nlp))

		X_train = single_aspect_reviews['reviewText']
		y_train = single_aspect_reviews['rating'].apply(lambda x: giveRating(x))
		print("✅ Data ready for model training.")

		# Step 6: Train model with cross-validation
		final_lr = Pipeline([
			('tfidf', TfidfVectorizer(lowercase=False, min_df=0.00006, ngram_range=(1,3))),
			('lr', LogisticRegression(solver='lbfgs', max_iter=175))
		])

		print("\n📈 Starting 5-fold cross-validation...")
		scores_final_lr = cross_validate(final_lr, X_train, y_train, scoring=scoring, cv=5)
		print("✅ Cross-validation complete. Results:")
		for scoring_measure, scores_arr in scores_final_lr.items():
			print(f"{scoring_measure}:\t{scores_arr.mean():.4f} (+/- {scores_arr.std() * 2:.4f})")

		# Step 7: Final model training
		print("\n🔧 Fitting final model on full data...")
		final_lr.fit(X_train, y_train)
		print("✅ Model trained.")

		print("\n💾 Saving model to disk...")
		dump(final_lr, 'models/model.joblib')
		print("✅ Model saved as 'models/model.joblib'.")

		model = final_lr

	return model
