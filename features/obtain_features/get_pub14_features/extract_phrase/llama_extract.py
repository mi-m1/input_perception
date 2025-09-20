import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm
import torch
import os

tqdm.pandas()
# Define your extraction function
def extract_referenced_term(sentence):
    
    messages = [
    {"role": "system", "content": "You are a useful data extraction assistant."},
    {"role": "user", "content":  f"Extract the term being referenced in the sentence below:\n\nSentence: {sentence}\n\n. Do not say anything else but the referenced term. Referenced Term:"},
    ]

    outputs = generator(
    messages,
    max_new_tokens=256,
    )

    # print(f"outptuts: {outputs}")
    response = outputs[0]["generated_text"][-1]
    # print(f"Response: {response}")

    print(f"referenced_term: {response['content'].split('Referenced Term:')[-1]}")

    referenced_term = response['content'].split('Referenced Term:')[-1]
    return referenced_term


# metonymy_path = "/mnt/parscratch/users/acq22zm/surprisal/dataset/pub_14_metonymy/pub14.csv"

# df = pd.read_csv(metonymy_path)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_name = "meta-llama/Llama-3.1-8B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
# generator = pipeline("text-generation", model=model, tokenizer=tokenizer,)

# Apply the extraction function
# df["referenced_term"] = df["pretext"].progress_apply(lambda x: extract_referenced_term(x))


save_path = "/mnt/parscratch/users/acq22zm/surprisal/dataset/pub_14_metonymy/pub14_extracted.csv"
df = pd.read_csv(save_path)

df["referenced_term"] = df["referenced_term"].progress_apply(lambda x: x.lower())
df.to_csv(save_path, index=False)
