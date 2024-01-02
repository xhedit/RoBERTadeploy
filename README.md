# RoBERTadeploy

## Forked from BERTdeploy


This is a small app I built using HuggingFace Transformers and FastAPI to perform text classification using the pre-trained DistilBERT model. I mostly relied on the excellent tutorial by Venelin to build this (ref 1). I made a few key changes to his approach:  

* Used pre-trained model instead of fine-tuning
* Used `requirement.txt` for pip instead of using pipenv
* Did not use a lot of extra code style packages

**How to use?**

* install torch for your hardware
* `pip install requirements.txt`
* `uvicorn DistilRoBERTa.api:app` or `bash bin/run_server`

Then make your API call:

```bash
http POST http://127.0.0.1:8000/classify text="Pre-trained j-hartmann/emotion-english-distilroberta-base seems to work quite well!"
```

You'll get an output like:


```js
{
    "probabilities": {
        "anger": 0.007748342119157314,
        "disgust": 0.0022821975871920586,
        "fear": 0.0021107119973748922,
        "joy": 0.27118009328842163,
        "neutral": 0.6292678713798523,
        "sadness": 0.005368099547922611,
        "surprise": 0.08204267174005508
    }
}
```

## TO-DO

* docker
* ??


## References

* [BERTdeploy](https://github.com/sshkhr/BERTdeploy)
* [Deploy BERT for Sentiment Analysis as REST API using PyTorch, Transformers by Hugging Face and FastAPI](https://curiousily.com/posts/deploy-bert-for-sentiment-analysis-as-rest-api-using-pytorch-transformers-by-hugging-face-and-fastapi/)
* [Auto-deploy FastAPI App to Heroku via Git in these 5 Easy Steps](https://towardsdatascience.com/autodeploy-fastapi-app-to-heroku-via-git-in-these-5-easy-steps-8c7958ef5d41)
