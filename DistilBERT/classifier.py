# run with "uvicorn DistilRoBERTa.api:app"

# config not used now

#import json
#with open("config.json") as json_file:
#	config = json.load(json_file)

#from transformers import AutoTokenizer, AutoModelForSequenceClassification
#import torch
#import torch.nn.functional as F

from transformers import pipeline
import pprint


class DistilRoBERTaClassifier():

	def __init__(self):
		self.pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

	# not working right now
	def batch_predict(self, texts, req_ids):

		pp = pprint.PrettyPrinter()

		result = self.pipeline(texts)
		print("result")
		pp.pprint(result)

		scores = []
		for i, outerlst in enumerate(result):
			current_scores = {}
			for score_dict in outerlst:

				classification = score_dict["label"]
				current_scores[classification] = score_dict["score"]
				print("label")
				print(classification)
				print("score")
				print(current_scores[classification])
			scores["req_id"] = req_ids[i]
			scores.append(current_scores)

		print("scores")
		pp.pprint(scores)

		return scores

	def predict(self, text, req_id):

		result = self.pipeline(text)

		scores = {}
		for outerlst in result:
			for score_dict in outerlst:
				classification = score_dict["label"]
				scores[classification] = score_dict["score"]
		return (req_id, scores)

berta = DistilRoBERTaClassifier()

def get_bert():
	return berta
