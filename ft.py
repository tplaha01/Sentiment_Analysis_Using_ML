import os
import fasttext

def get_model():
	model_path = "models/fasttext_model_cbow.bin"
	data_path = "csv_files/fasttext_data.txt"

	# Ensure models/ directory exists
	os.makedirs(os.path.dirname(model_path), exist_ok=True)

	if os.path.isfile(model_path):
		print("FastText model already exists")
		model = fasttext.load_model(model_path)
	else:
		print("FastText model doesn't exist. Training and saving it.")

		# Check if data file exists and is not empty
		if not os.path.isfile(data_path) or os.path.getsize(data_path) == 0:
			raise ValueError(f"Training data file '{data_path}' is missing or empty.")

		model = fasttext.train_unsupervised(data_path, model='cbow')
		model.save_model(model_path)
	
	return model
