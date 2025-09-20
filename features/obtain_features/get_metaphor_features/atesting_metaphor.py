# use this for: surprisal,entropy, cws, cis...etc.
import os, sys
# here = os.path.dirname(os.path.abspath(__file__))

# project_root = os.path.abspath(os.path.join(here, os.pardir, os.pardir))
# sys.path.insert(0, project_root)
import pandas as pd
from tqdm import tqdm
import re
from statistics import mean, median, stdev
import numpy
from utils import *

sys.path.append(("../../.."))  # Add src to path
from src.prompt_gpts import *
from src.surprisal_iterative import *
from src.load_models_and_tokeniser import * 


import statistics
import argparse


class Extractor:

    def __init__(self, tokenizer, query, cleaned_sentence, extractedPhrase, calc_func, start_of_sentence_token, model, gamma,):

        self.tokenizer = tokenizer
        # self.model_name = model_name
        self.query = query
        self.model = model
        self.start_of_sentence_token = start_of_sentence_token

        #calc_func = surprisal calculation function, for example
        self.calc_func = calc_func
        self.gamma = gamma

        self.query_tokens = tokenizer(query, add_special_tokens=False).tokens()[:] #no first token now, take all

        self.sentence_tokens = tokenizer(f" {cleaned_sentence}", add_special_tokens=False).tokens()[:] #no first token now, take all
        self.sentence_tokens.append(start_of_sentence_token) 
        
        self.extractedPhrase = extractedPhrase
        print("-----------------------------------------")
        print(f"** {self.calc_func.__name__} **")
        print(f"phrase: {extractedPhrase}")
        self.extractedPhrase_tokens = tokenizer(" " + extractedPhrase, add_special_tokens=False).tokens()[:] #no first token now, take all


        if cleaned_sentence.startswith(extractedPhrase):
            print("Phrase at the start of the sentence")
            self.extractedPhrase_tokens = tokenizer(" " + extractedPhrase, add_special_tokens=False).tokens()[:] #no first token now, take all
        
            # extractedPhrase_tokens = encoding_phrase.tokens()[1:] # Remove the first token, which are "BOS"
            print(f"\tTokens of phrase: {self.extractedPhrase_tokens}")

        elif (extractedPhrase.startswith("‘") and extractedPhrase.endswith("’")) or (extractedPhrase.startswith('"') and extractedPhrase.endswith('"')) or (extractedPhrase.startswith("'") and extractedPhrase.endswith("'")):
            print("Phrase is in quotes")
            encoding_phrase = tokenizer(" "+ extractedPhrase, add_special_tokens=False)
            extractedPhrase_tokens = encoding_phrase.tokens()[1:-1] # remove the first and last quotes (no first token now)
            self.extractedPhrase_tokens = extractedPhrase_tokens
            print(f"\tTokens of phrase: {self.extractedPhrase_tokens}")

        elif (extractedPhrase.startswith('(') and extractedPhrase.endswith(")")):
            print("Phrase is surronded by brackets")
            encoding_phrase = tokenizer(" " + extractedPhrase + " ", add_special_tokens=False)
            self.extractedPhrase_tokens = encoding_phrase.tokens()[1:-2]
            
            print(f"\tTokens of phrase: {self.extractedPhrase_tokens}")

        elif extractedPhrase.startswith("‘") or extractedPhrase.startswith('('):
            print("Phrase starts with a single quote or single bracket")
            encoding_phrase = tokenizer(" " + extractedPhrase, add_special_tokens=False)
            self.extractedPhrase_tokens = encoding_phrase.tokens()[1:]
            print(f"\tTokens of phrase: {self.extractedPhrase_tokens}")

        elif extractedPhrase.startswith("-"):
            print(f"Phrase starts with -")
            encoding_phrase = tokenizer(extractedPhrase, add_special_tokens=False)
            self.extractedPhrase_tokens = encoding_phrase.tokens()[1:]
            print(f"\tTokens of phrase: {self.extractedPhrase_tokens}")


        else:
            print("Phrase not at the start of the sentence")
            encoding_phrase = tokenizer("poly " + extractedPhrase, add_special_tokens=False)     
        
            self.extractedPhrase_tokens = encoding_phrase.tokens()[1:] # Remove the first token, which is "poly"
            print(f"\tTokens of phrase: {self.extractedPhrase_tokens}")

    def values_for_whole_query(self, *args, **kwargs):
        # run calc_func function on the self.query

        kwargs.setdefault('sentence', self.query)
        kwargs.setdefault('tokenizer', self.tokenizer)
        kwargs.setdefault('model', self.model)

        if self.calc_func.__name__ == "cws" or self.calc_func.__name__ == "cis":
            _, values_for_query = self.calc_func(*args, **kwargs,)
        else:
            values_for_query = self.calc_func(*args, **kwargs)
        
        self.values_for_whole_query = values_for_query
        return values_for_query
    
    def find_sentence_in_query(self, values_for_query,):
        #find sentence index
        print(f"sentence_tokens: {self.sentence_tokens}")
        sentence_index_in_query = find_tuple_positions(values_for_query, self.sentence_tokens)
        print(f"Sentence index in query: {sentence_index_in_query}")
        self.sentence_index_in_query = sentence_index_in_query

        values_for_sentence = values_for_query[sentence_index_in_query[0]:sentence_index_in_query[1]+1]
        self.values_for_sentence = values_for_sentence

        return values_for_sentence
    
    def find_phrase_index(self,):

        phrase_index = find_subsequence_positions(self.values_for_sentence, self.extractedPhrase_tokens)

        # check the position indexes aren't empty
        if phrase_index != (None, None):
            print("Safe!")
            self.phrase_index = phrase_index

            # #check if the positions are the same - they should be!
            # # so we can use either as the index to get the values
            # if phrase_positions_cws == phrase_positions_surprisal:
            #             phrase_positions = phrase_positions_cws
            # else:
            #     raise Exception("Positions of phrase in cws_values and surprisal_values are not the same.")

        else:
            print("Not safe!")

        return phrase_index
    
    def find_expression_in_sentence(self,values_for_sentence):
        
        values_for_phrase = values_for_sentence[self.phrase_index[0]:self.phrase_index[1]+1]
        self.values_for_phrase = values_for_phrase
        
        return values_for_phrase

    def find_context_in_sentence(self, values_for_sentence,):

        values_for_context = take_context(values_for_sentence, self.phrase_index)
        self.values_for_context = values_for_context

        return values_for_context


def get_list_from_list_of_tuples(list_of_tuples):
        flat_list = [float(item[1]) for item in list_of_tuples]

        return flat_list



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
                'std': numpy.std(values),
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
    parser.add_argument("--path_to_dice", type=str, help="Path to DICE dataset")    
    parser.add_argument("--hf_model", type=str, help="Hugging Face model handle")
    parser.add_argument("--cws_gamma", type=float, help="Gamma value for CWS (default = 0.5)", default=0.5)
    parser.add_argument("--task", type=str, help="Task to run: 'MOH-X' or 'TroFi'",)
    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = process_options()

    all_results = []


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model_name = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    model = AutoModelForCausalLM.from_pretrained(args.hf_model).to(device)
    model.eval()

    gamma = 0.5

    df = pd.read_csv(args.path_to_dice)
    # df = pd.read_csv(args.path_to_dice).iloc[344:400]

    idiom_sent_occurrence= list(zip(df.ID, df.target_word, df.sentence, df.Extracted_Phrase, df.label))

    if "google" in args.hf_model:
        start_of_sentence_token = "▁Answer"
    else:
        start_of_sentence_token = "ĠAnswer"

    metrics = [
            surprisal,
            cws,
            vanilla_entropy,
            cis,
            ]
    

    # start of the loop
    for id, base_exp, cleaned_sentence, extractedPhrase, label in tqdm(idiom_sent_occurrence):

        features_for_metric = []

        for metric in metrics:

            # Initialize Extractor
            extractor = Extractor(
                tokenizer=tokenizer,
                # model_name="mock-model",
                query=f"Is the word '{base_exp}' used metaphorically or literally in the sentence: {cleaned_sentence}. Answer 'i' for metaphorical, 'l' for literal.  Put your answer after 'output: '.",
                # query=f"Is the expression '{base_exp}' used figuratively or literally in the sentence: {cleaned_sentence} Answer 'i' for figurative, 'l' for literal.  Put your answer after 'output: '.",
                # calc_func=surprisal,
                calc_func=metric,
                cleaned_sentence=cleaned_sentence,
                extractedPhrase=extractedPhrase,
                start_of_sentence_token=start_of_sentence_token,
                model=model,
                gamma=args.cws_gamma, #for cws
            )

            # Run the pipeline


            query_values = extractor.values_for_whole_query()
            print(f"Query values: {query_values}")
            sentence_values = extractor.find_sentence_in_query(query_values)
            print(f"Sentence_values: {sentence_values}")

            phrase_index = extractor.find_phrase_index() # relative to sentence_values
            print(f"Phrase index: {phrase_index}")

            phrase_values = extractor.find_expression_in_sentence(sentence_values)
            print(f"Phrase_values: {phrase_values}")

            context_values = extractor.find_context_in_sentence(sentence_values)
            print(f"context values: {context_values}")



            ###################### FEATURES ######################
            # Compute statistics for each scope

            data_dictionary = {
                "sentence":sentence_values,
                "query":query_values,
                "phrase":phrase_values,
                "context":context_values,
            }

            features_calculator = Features_Calculator(
                data_raw=data_dictionary,
                phrase_index=phrase_index,
                metric_name=extractor.calc_func.__name__,
                )
            
            basic_stats_info = features_calculator.basic_stats()

            print(f"basic_stats_info:{basic_stats_info}")

            boundary_dict = features_calculator.boundary()

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
            row = {
                "id": id,
                "idiom": base_exp,
                "sentence": cleaned_sentence,
                "extracted_phrase": extractedPhrase,
                "label": label,
                # "metric": metric_func.__name__,
            }
            row.update(flattened_basic_stats)
            row.update(boundary_features)
            row.update(positional_features)
            row.update(surprisal_drop)
            row.update(surprisal_spikes)
            row.update(relational_values)

            # print(f"row:\n-----\n{row}")

            features_for_metric.append(row)

        # Join up all features for a idiom to be a row (i.e., a big dictionary inside all_results)
        big_dict = {k: v for d in features_for_metric for k, v in d.items()}

        print(f"features_for_metric:\n{big_dict}")
        # print(f"{type(features_for_metric)}")
        all_results.append(big_dict)

        # break

    
    #write all to .csv
    path_to_save = f"../features/{args.task}/{args.hf_model}/"
    os.makedirs(path_to_save, exist_ok=True)

    df_to_save = pd.DataFrame(all_results)
    # print(f"df_to_save:\n{df_to_save.shape}")

    df_to_save.to_csv(f"{path_to_save}cws_y{gamma}.csv", index=False)



