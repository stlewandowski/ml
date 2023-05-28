from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, TFDistilBertForSequenceClassification
from transformers import pipeline
from datetime import datetime
# use huggingface's sentiment analysis pipeline to analyze the text
# https://huggingface.co/transformers/main_classes/pipelines.html#transformers.pipeline

def analyze_text(text):
    # create the pipeline
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
    model = TFDistilBertForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    # analyze the text
    result = nlp(text)
    # return the result
    return result

if __name__ == "__main__":
    dtn = datetime.now()
    print(dtn)
    at = analyze_text("US shale explorers are well-prepared to manage a potential credit crisis after piling up cash and paying down debt,… https://t.co/Ei4pgZveAd")
    dtt = datetime.now()
    print(dtt)
    print(dtt - dtn)
    print(at)


