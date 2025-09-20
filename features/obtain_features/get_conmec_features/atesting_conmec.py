from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
import torch
import os
import argparse
from tqdm import tqdm
from statistics import mean, median, stdev
from utils import *
import sys

sys.path.append(("../../.."))  # Add src to path
from src.prompt_gpts import *
from src.surprisal_iterative import *
from src.load_models_and_tokeniser import * 



def surprisal(tokenizer, model, sentence,):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = model.to(device)
    model.eval()

    tokens = tokenizer.tokenize(sentence, return_tensors="pt")
    print(f"tokens (surprisal_func): {tokens}")

    # Dynamically check if the BOS token exists and build input_ids accordingly
    if tokenizer.bos_token is not None:
        input_ids = torch.tensor(
            tokenizer.encode(tokenizer.bos_token) + tokenizer.encode(sentence)
        ).unsqueeze(0).to(device)
    else:
        input_ids = torch.tensor(
            tokenizer.encode(sentence)
        ).unsqueeze(0).to(device)

    with torch.no_grad():
      logits = model(input_ids)[0].to(torch.float64) #logits

    # (batch_size, sequence_length, config.vocab_size)
    # print(f"logits.shape: {logits.shape}") #torch.Size([1, 8, 50257])

    scores = []
    for index in range(input_ids.shape[1]):

        # print(f"index:{index}")
        # scores.append(str(-1 * math.log2(torch.nn.Softmax(0)(logits[0][index-1])[input_ids[0][index]].item())))
        scores.append(str(-1 * np.log2(torch.nn.Softmax(dim=0)(logits[0][index-1])[input_ids[0][index]].item())))

    surprisal_per_token = list(zip(tokens,scores[1:len(scores)]))

    return surprisal_per_token

def get_token_indices_for_substring(tokenizer, sentence, substring):
    '''
    Given a sentence and a substring, this function returns the token positions
    e.g., [3,4,5] means the substring positioned as 3rd, 4th, and 5th token in the sentence
    Note: this range is in line with python (0-indexing)
    '''
    # Find character positions of the substring
    start_char = sentence.find(substring)
    print(f"start_char:{start_char}")

    if start_char == -1:
        raise ValueError(f"Substring '{substring}' not found in sentence")
    end_char = start_char + len(substring)
    print(f"end_char:{end_char}")

    # Tokenize the sentence and get character-to-token mapping
    encoding = tokenizer(sentence, return_offsets_mapping=True, add_special_tokens=False)
    # encoding = tokenizer(sentence, return_offsets_mapping=True,)
    tokens = encoding.tokens()
    print(f"tokens (find func): {tokens}")
    offsets = encoding['offset_mapping']

    # Find tokens that overlap with the substring
    token_indices = []
    for i, (start, end) in enumerate(offsets):
        # Check if token overlaps with substring
        if end > start_char and start < end_char:
            token_indices.append(i)

    return token_indices, tokens

# base_exp = "spilled the beans"
# cleaned_sentence = "He accidentally spilled the beans about the surprise party."
# query=f"Is the word '{base_exp}' used metaphorically or literally in the sentence: {cleaned_sentence} Answer 'i' for metaphorical, 'l' for literal.  Put your answer after 'output: '."

# sentence_token_indices, tokens = get_token_indices_for_substring(tokenizer, sentence=query, substring=cleaned_sentence)
# print(f"sentence Token indices: {sentence_token_indices}")
# print(f"Corresponding tokens: {[tokens[i] for i in sentence_token_indices]}")


# phrase_token_indices, tokens = get_token_indices_for_substring(tokenizer, sentence=cleaned_sentence, substring=base_exp)
# print(f"phrase Token indices: {phrase_token_indices}")
# print(f"Corresponding tokens: {[tokens[i] for i in phrase_token_indices]}")


# query_values = surprisal(tokenizer, model, sentence=query)
# sentence_values = query_values[sentence_token_indices[0]-1:sentence_token_indices[-1]]
# print(f"sentence_surprisal: {sentence_values}")
# phrase_values = sentence_values[phrase_token_indices[0]-1:phrase_token_indices[-1]]
# print(f"phrase_surprisal: {phrase_values}")

def take_context(my_list, index_tuple):
    start = index_tuple[0]
    end = index_tuple[1]
    return my_list[:start] + my_list[end:]


def checker(token_value_tuple, dataset_string):
    # check if extracted phrase == extracted phrase in dataset
    tokens = [tok for tok, _ in token_value_tuple]
    reconstructed_text = tokenizer.convert_tokens_to_string(tokens)

    if reconstructed_text.strip() == dataset_string.strip():
        return True
    else:
        print(f"Bad! {reconstructed_text} != {dataset_string}")
        return False


def checker_context(context_values, phrase_values, dataset_sentence_cleaned,):
    # check if extracted phrase == extracted phrase in dataset
    tokens = [tok for tok, _ in phrase_values]
    reconstructed_phrase = tokenizer.convert_tokens_to_string(tokens)

    context_tokens = [tok for tok, _ in context_values]
    reconstructed_context = tokenizer.convert_tokens_to_string(context_tokens)

    # add space to the beginning of the dataset cleaned
    # because to ensure that the tokenised phrase matches how the sentence apears in the query
    # space is added to the beginning of the sentence (i.e., sentence: {sentence_cleaned})
    dataset_sentence_cleaned = " "+ dataset_sentence_cleaned

    if reconstructed_phrase in dataset_sentence_cleaned:
        result = dataset_sentence_cleaned.replace(reconstructed_phrase, "", 1)  # Remove only the first occurrence
        return result == reconstructed_context
    return False
    

class Features_Calculator:
    def __init__(self, data_raw, phrase_index, metric_name):
        # data could be any iterable (list, numpy array, etc.)
        self.data_raw = data_raw

        self.func_name = metric_name

        self.data_dict = {}
        for level, list_of_tuples in data_raw.items():
            
            list_of_values = get_list_from_list_of_tuples(list_of_tuples)

            self.data_dict[level] = list_of_values

        self.phrase_index = phrase_index

        # print(f"self.data_dict:{self.data_dict}")


    def basic_stats(self):

        basic_stats_scores = {}

        for level, values in self.data_dict.items():

            scores = {
                'mean': mean(values),
                'std': np.std(values),
                'min': min(values),
                'max': max(values),
                'sum': sum(values),
                # 'count': values.size,
                'middle': find_middle(values),
                'median': median(values),
            }

            basic_stats_scores[f"{self.func_name}_{level}"] = scores

        return basic_stats_scores


    def boundary(self):
        '''
        - Mean(1st word of expression, last_word of expression)
        - Diff (last_word of expression - first word of expression)
        - Mean (last token of expression position, first word after expression)
        - Diff (first word after expression - last token of expression position)
        '''
        #needs phrase-level list
        phrase_head_and_tail_avg = mean([self.data_dict["phrase"][0], self.data_dict["phrase"][-1]])

        #delta = last_word in expression - first word in expression
        phrase_head_and_tail_diff = self.data_dict["phrase"][-1] - self.data_dict["phrase"][0] 

                #last token of expression position
        position_token_at_end_of_expression = self.phrase_index[-1]
        position_first_word_after_expression = position_token_at_end_of_expression + 1

        sentence_values = self.data_dict["sentence"]


        print(f"last word of idiom: {self.data_raw["sentence"][position_token_at_end_of_expression]}")
        print(f"first word after idiom: {self.data_raw["sentence"][position_first_word_after_expression]}")

        print(f"value of last word of expression: {sentence_values[position_token_at_end_of_expression]}")
        print(f"value of first word after the expression: {sentence_values[position_first_word_after_expression]}")

        boundary_phraseEnd_firstWordAfter_mean= mean([sentence_values[position_token_at_end_of_expression], sentence_values[position_first_word_after_expression]])
        boundary_phraseEnd_firstWordAfter_diff = sentence_values[position_first_word_after_expression] - sentence_values[position_token_at_end_of_expression]

        dict_to_return = {
            f"{self.func_name}_phrase_head_and_tail_avg": phrase_head_and_tail_avg,
            f"{self.func_name}_phrase_head_and_tail_diff": phrase_head_and_tail_diff,
            f"{self.func_name}_boundary_phraseEnd_firstWordAfter_mean": boundary_phraseEnd_firstWordAfter_mean,
            f"{self.func_name}_boundary_phraseEnd_firstWordAfter_diff": boundary_phraseEnd_firstWordAfter_diff,
        } 

        return dict_to_return

    def positional_features(self):

        max_val = max(self.data_dict["sentence"])
        min_val = min(self.data_dict["sentence"])

        #positional of max and min values in the sentence
        position_sentenceMax = float(self.data_dict["sentence"].index(max_val))
        position_sentenceMin = float(self.data_dict["sentence"].index(min_val))

        # check if the max, min surprisal token is in the phrase or context
        # if max (or min) is in the phrase, = 1.0, else 0.0 meaning it's in the context.
        positional_sentenceMax_in_expression= float(max_val in self.data_dict["phrase"])
        positional_sentenceMin_in_expression= float(min_val in self.data_dict["phrase"])

        dict_to_return = {
            f"{self.func_name}_position_sentenceMax": position_sentenceMax,
            f"{self.func_name}_position_sentenceMin": position_sentenceMin,
            f"{self.func_name}_positional_sentenceMax_in_expression": positional_sentenceMax_in_expression, 
            f"{self.func_name}_positional_sentenceMin_in_expression": positional_sentenceMax_in_expression,
        }

        return dict_to_return

    @staticmethod
    def check_if_list_is_decreasing(ls):
        """Retursn 1.0 if the list is decreasing, 0.0 otherwise."""
        return float(all(earlier >= later for earlier, later in zip(ls, ls[1:])))

    def surprisal_drop_within_expression(self,):
        dict_to_return = {
            f"{self.func_name}_drops_within_expression": self.check_if_list_is_decreasing(self.data_dict["phrase"]),
        }
        return dict_to_return
    

    @staticmethod
    def count_spikes(lst):
        """Count the number of spikes in the list."""
        spike_count = 0
        for i in range(1, len(lst) - 1):
            if lst[i] > lst[i - 1] and lst[i] > lst[i + 1]:
                spike_count += 1  # Increment count when a spike is found
        return spike_count
    

    def surprisal_spikes_within_expression(self):
        spikes_phrase = float(self.count_spikes(self.data_dict["phrase"]))

        dict_to_return = {
            f"{self.func_name}_spikes_within_expression": spikes_phrase,
        }
        return dict_to_return
    
    def relational_values(self,):
        relational_sumPhrase_sumSentence = safe_division(sum(self.data_dict["phrase"]), sum(self.data_dict["sentence"]))
        relational_maxPhrase_maxSentence = safe_division(max(self.data_dict["phrase"]), max(self.data_dict["sentence"]))
        relational_minPhrase_minSentence = safe_division(min(self.data_dict["phrase"]), min(self.data_dict["sentence"]))

        dict_to_return = {
            f"{self.func_name}_relational_sumPhrase_sumSentence": relational_sumPhrase_sumSentence,
            f"{self.func_name}_relational_maxPhrase_maxSentence": relational_maxPhrase_maxSentence,
            f"{self.func_name}_relational_minPhrase_minSentence": relational_minPhrase_minSentence,
        }

        return dict_to_return

def process_options():
    parser = argparse.ArgumentParser(description="Code for running surprisal and cws on DICE dataset")
    parser.add_argument("--path_to_data", type=str, help="Path to DICE dataset")    
    parser.add_argument("--hf_model", type=str, help="Hugging Face model handle")
    parser.add_argument("--cws_gamma", type=float, help="Gamma value for CWS (default = 0.5)", default=0.5)
    parser.add_argument("--task", type=str, help="Task to run: 'MOH-X' or 'TroFi'",)
    args = parser.parse_args()

    return args



if __name__ == "__main__":
    args = process_options()

    model_name = args.hf_model

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")


    metrics = [
        surprisal,
        cws,
        vanilla_entropy,
        cis,
    ]


    # df = pd.read_csv("/mnt/parscratch/users/acq22zm/surprisal/dataset/conmec/conmec_balanced.csv")
    df = pd.read_csv(args.path_to_data)

    print(f"columns:{df.columns.to_list()}")
    print('Target Word (Noun)' in df.columns)  # Check specific column
    print(f"{df['Target Word (Noun)'].to_list()}")
    all_results = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        print(row["Target Word (Noun)"])

        features_for_metric = []

        print(f"-----{idx}--------")

        for calc_func in metrics:


    #         # print(f"row:\n{row}")
            
            base_exp = row['Target Word (Noun)']
            print(f"base_exp: {base_exp}")
            # cleaned_sentence = row['Sentence_Cleaned']
            cleaned_sentence = row['Sentence']
            # extracted_exp = row['Extracted_Phrase']
            extracted_exp = base_exp

            query=f"Is the word '{base_exp}' used metonymically or literally in the sentence: {cleaned_sentence} Answer 'i' for metonymic, 'l' for literal.  Put your answer after 'output: '."

            sentence_token_indices, tokens = get_token_indices_for_substring(tokenizer, sentence=query, substring=cleaned_sentence)
            print(f"sentence Token indices: {sentence_token_indices}")
            print(f"Corresponding tokens: {[tokens[i] for i in sentence_token_indices]}")

            phrase_token_indices, tokens = get_token_indices_for_substring(tokenizer, sentence=" "+ cleaned_sentence, substring=extracted_exp)
            # phrase_token_indices, tokens = get_token_indices_for_substring(tokenizer, sentence=cleaned_sentence, substring=extracted_exp)

            print(f"phrase Token indices: {phrase_token_indices}")
            print(f"Corresponding tokens: {[tokens[i] for i in phrase_token_indices]}")


            # phrase_start = phrase_token_indices[0], phrase_token_indices[-1]+1]

            if calc_func.__name__ == "cws" or calc_func.__name__ == "cis":
                _, query_values = calc_func(tokenizer, model, sentence=query)
            else:
                query_values = calc_func(tokenizer, model, sentence=query)

            if calc_func.__name__ == "cis":
                phrase_token_indices[0] = phrase_token_indices[0] - 1 

            print(f"query_{calc_func.__name__}: {query_values}")
            sentence_values = query_values[sentence_token_indices[0]:sentence_token_indices[-1]+1]
            print(f"sentence_{calc_func.__name__}: {sentence_values}")

            phrase_values = sentence_values[phrase_token_indices[0]:phrase_token_indices[-1]+1]
            print(f"phrase_{calc_func.__name__}: {phrase_values}")
            
            context_values = take_context(sentence_values, [phrase_token_indices[0], phrase_token_indices[-1]+1])
            print(f"context_{calc_func.__name__}: {context_values}")

            print(f"\t** QUALITY CHECKS -- {calc_func.__name__} **")
            sentence_check = checker(sentence_values, row["Sentence"])
            phrase_check = checker(phrase_values, row["Target Word (Noun)"])
            context_check = checker_context(context_values, phrase_values, row["Sentence"])

            print(f"\tsentence_check: {sentence_check}")
            print(f"\tphrase_check: {phrase_check}")
            print(f"\tcontext_check: {context_check}")
            print(f"\t****************")

            data_dictionary = {
                "sentence":sentence_values,
                "query":query_values,
                "phrase":phrase_values,
                "context":context_values,
            }

            print(f"data_dictionary: {data_dictionary}")

            features_calculator = Features_Calculator(
                data_raw=data_dictionary,
                phrase_index=phrase_token_indices,
                metric_name=calc_func.__name__,
                )
            
            # Run the methods to get feature outputs.
            basic_stats = features_calculator.basic_stats()              # This is a nested dictionary.
            boundary_features = features_calculator.boundary()             # Dictionary output.
            positional_features = features_calculator.positional_features()  # Dictionary output.
            surprisal_drop = features_calculator.surprisal_drop_within_expression()  # Dictionary output.
            surprisal_spikes = features_calculator.surprisal_spikes_within_expression()  # Dictionary output.
            relational_values = features_calculator.relational_values()    # Dictionary output.

            # Flatten the nested basic_stats dictionary.
            flattened_basic_stats = {}

            for level, stats in basic_stats.items():
                for stat_name, value in stats.items():
                    # Create a new key that combines the level and the stat name.
                    flattened_basic_stats[f"{level}_{stat_name}"] = value
            
            print(f"flattened_basic_stats: {flattened_basic_stats}")
            
            # Combine all features into one row
            save_row = {
                "label": row["Label"],
                "expression": base_exp,
                "sentence": cleaned_sentence,
                "extracted_phrase": base_exp,
                "metric": calc_func.__name__,
                "id": row["Document URL"],
            }

            save_row.update(flattened_basic_stats)
            save_row.update(boundary_features)
            save_row.update(positional_features)
            save_row.update(surprisal_drop)
            save_row.update(surprisal_spikes)
            save_row.update(relational_values)

            # print(f"row:\n-----\n{row}")

            features_for_metric.append(save_row)

        print(f'finished row: {idx}')

        # Join up all features for a idiom to be a row (i.e., a big dictionary inside all_results)
        big_dict = {k: v for d in features_for_metric for k, v in d.items()}

        print(f"features_for_metric:\n{big_dict}")
        # print(f"{type(features_for_metric)}")
        all_results.append(big_dict)


    #write all to .csv
    path_to_save = f"../features/{args.task}/{args.hf_model}/"
    os.makedirs(path_to_save, exist_ok=True)

    df_to_save = pd.DataFrame(all_results)
    # print(f"df_to_save:\n{df_to_save.shape}")

    df_to_save.to_csv(f"{path_to_save}cws_y{args.cws_gamma}.csv", index=False)


    #     break
    # break
                                
    # print(f"-------------")                      
    # cleaned_sentence = "Nancy Reagan codified the misty-eyed gaze at the rugged man, the demure demurrals, and the aggregation of power behind the throne, while claiming, in interviews, interest in nothing more serious than the White House’s latest china patterns."
    # extracted_phrase = "power behind the throne"
    # query=f"Is the word '{extracted_phrase}' used metaphorically or literally in the sentence: {cleaned_sentence} Answer 'i' for metaphorical, 'l' for literal.  Put your answer after 'output: '."

    # sentence_token_indices, tokens = get_token_indices_for_substring(tokenizer, sentence=query, substring=cleaned_sentence)
    # print(f"sentence Token indices: {sentence_token_indices}")
    # print(f"Corresponding tokens: {[tokens[i] for i in sentence_token_indices]}")


    # phrase_token_indices, tokens = get_token_indices_for_substring(tokenizer, sentence=" "+cleaned_sentence, substring=extracted_phrase)
    # print(f"phrase Token indices: {phrase_token_indices}")
    # print(f"Corresponding tokens: {[tokens[i] for i in phrase_token_indices]}")


    # query_values = surprisal(tokenizer, model, sentence=query)
    # sentence_values = query_values[sentence_token_indices[0]:sentence_token_indices[-1]+1]
    # print(f"sentence_surprisal: {sentence_values}")
    # phrase_values = sentence_values[phrase_token_indices[0]:phrase_token_indices[-1]+1]
    # print(f"phrase_surprisal: {phrase_values}")
