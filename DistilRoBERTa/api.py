from typing import Dict

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from .classifier import DistilRoBERTaClassifier, get_bert

app = FastAPI()

class ClassificationRequest(BaseModel):
	text: str
	req_id: str

class ClassificationResponse(BaseModel):
	probabilities: Dict[str, float]
	req_id: str

class BatchClassificationRequest(BaseModel):
	texts: list[str]
	req_ids: list[str]

class BatchClassificationResponse(BaseModel):
	probabilities: list[Dict[str, float]]
	req_ids: list[str]


@app.post("/classify", response_model = ClassificationResponse)
def classify(request: ClassificationRequest, model: DistilRoBERTaClassifier = Depends(get_bert)):
	req_id, probabilities = model.predict(request.text, request.req_id)
	return ClassificationResponse(
		probabilities = probabilities,
		req_id = req_id
	)

@app.post("/batch_classify", response_model = BatchClassificationResponse)
def classify(request: BatchClassificationRequest, model: DistilRoBERTaClassifier = Depends(get_bert)):
	req_ids, probabilities = model.batch_predict(request.texts, request.req_ids)
	return ClassificationResponse(
		probabilities = probabilities,
		req_ids = req_ids
	)

