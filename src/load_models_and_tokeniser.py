
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM


def hf_load_model_and_tokeniser(model_name, api):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
            model_name, #="meta-llama/Llama-2-7b-hf"
            device_map="auto", 
            torch_dtype="auto",
            token = api,
        )

    return tokenizer, model

def hf_load_model_and_tokeniser_seq2seq(model_name, api):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, #="meta-llama/Llama-2-7b-hf"
            device_map="auto", 
            torch_dtype="auto",
            token = api,
        )

    return tokenizer, model


if __name__ == "__main__":

    tokeniser, model = hf_load_model_and_tokeniser_seq2seq(
        model_name = "google/flan-t5-xxl",
        api = "your_access_token",
    )

    tokens = tokeniser.tokenize('"agony aunt"')
    print(tokens)
    
    tokens = tokeniser.tokenize(' agony aunt ')
    print(tokens)

    tokens = tokeniser.tokenize('agony aunt')
    print(tokens)

    tokens = tokeniser.tokenize('" agony aunt "')
    print(tokens)

    tokens = tokeniser.tokenize(' spill the beans ')
    print(tokens)

    tokens = tokeniser.tokenize('spill the beans')
    print(tokens)





