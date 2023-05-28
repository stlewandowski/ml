from transformers import AutoModelForCausalLM, AutoTokenizer, TFAutoModelForCausalLM
from transformers import pipeline
from datetime import datetime
from pprint import pp


def pipeline_generate(prompt, length):
    print("pipeline_generate")
    b = datetime.now()
    # this uses pytorch
    nlp = pipeline("text-generation", model="gpt2-large")
    #nlp = pipeline("text-generation")
    res = nlp(prompt, max_length=length, num_return_sequences=1)
    a = datetime.now()
    print("Time to generate (in seconds): ", (a - b).total_seconds())
    pp(res)

def beam_search_decoding(prompt, length, checkpoint):
    print("beam_search_decoding")
    b = datetime.now()
    # this uses pytorch
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    inputs = tokenizer(prompt, return_tensors="pt")
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    outputs = model.generate(**inputs, num_beams=5, max_new_tokens=length)
    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    a = datetime.now()
    difference = a - b
    print("Time to generate (in seconds): ", difference.total_seconds())
    pp(res)

def multinomial_sampling(prompt, length, checkpoint):
    print("multinomial_sampling")
    b = datetime.now()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, do_sample=True, num_beams=1, max_new_tokens=length)
    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    a = datetime.now()
    print("Time to generate (in seconds): ", (a - b).total_seconds())
    pp(res)

def contrastive_search(prompt, length, checkpoint):
    print("contrastive_search")
    b = datetime.now()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, penalty_alpha=0.6, top_k=4, max_new_tokens=length)
    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    a = datetime.now()
    print("Time to generate (in seconds): ", (a - b).total_seconds())
    pp(res)

if __name__ == "__main__":
    prompt = "K3 was a large earth-like exoplanet discovered"
    checkpoint = "gpt2-large"
    length = 50
    beam_search_decoding(prompt, length, checkpoint)
    multinomial_sampling(prompt, length, checkpoint)
    contrastive_search(prompt, length, checkpoint)
    pipeline_generate(prompt, length)