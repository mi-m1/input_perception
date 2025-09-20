from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch
import math
import numpy as np
import re

# Check if GPU is available and set device


def softmax(x):
	return np.exp(x)/sum(np.exp(x))
    

def tokenise_expression(tokeniser, expression):
    tokens = tokeniser.tokenize(expression, return_tensors="pt")
    # print(tokens)
    return tokens


def clean_sentence(sentence,idiom):
    
    pattern = rf"(['\"]){re.escape(idiom)}\1"
    result_without_quotes = re.sub(pattern, rf"{idiom}", sentence)

    return result_without_quotes

def surprisal(tokenizer, model, sentence,):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = model.to(device)
    model.eval()

    tokens = tokenizer.tokenize(sentence, return_tensors="pt")

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



def surprisal_t5(tokenizer, model, sentence):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    tokens = tokenizer.tokenize(sentence, return_tensors="pt")

    #https://huggingface.co/docs/transformers/model_doc/t5

    input_ids = tokenizer(str(sentence), return_tensors="pt").input_ids.to(device)
    labels = tokenizer(str(sentence), return_tensors="pt").input_ids.to(device)

    # the forward function automatically creates the correct decoder_input_ids
    # loss = model(input_ids=input_ids, labels=labels).loss
    # print(f"loss: {loss}")

    logits = model(input_ids=input_ids, labels=labels).logits #Seq2SeqLMOutput
    # print(f"logits: {logits.shape}")

    scores = []
    for index in range(input_ids.shape[1]):

        scores.append(str(-1 * np.log2(torch.nn.Softmax(0)(logits[0][index-1])[input_ids[0][index]].item())))

    surprisal_per_token = list(zip(tokens,scores[1:len(scores)]))

    return surprisal_per_token

    # print(f"surprisal: {surprisal_per_token}")

def cws(tokenizer, model, sentence, gamma=0.5):

    """
    gamma: controls the influence of the KL Divergence - CWS=Surprisal+γ⋅KL Divergence
    TODO: think about implementing threshold?
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    model.eval()

    tokens = tokenizer.tokenize(sentence, return_tensors="pt")
    # print(tokens)

    # Dynamically check if the BOS token exists and build input_ids accordingly
    if tokenizer.bos_token is not None:
        input_ids = torch.tensor(
            tokenizer.encode(tokenizer.bos_token) + tokenizer.encode(sentence)
        ).unsqueeze(0).to(device)
    else:
        input_ids = torch.tensor(
            tokenizer.encode(sentence)
        ).unsqueeze(0).to(device)

    # input_ids = torch.tensor(tokenizer.encode(tokenizer.bos_token) + tokenizer.encode(sentence)).unsqueeze(0).to(device)

    # print(f"input_ids:\n{input_ids}")
    # breakpoint()

    #on cuda
    logits = model(input_ids)[0].to(torch.float64) #logits

    # things for KL-Divergence
    vocab_size = logits.size(-1)
    smooth_prob_mass = 0.1
    

    surprisal_scores = []
    cws_scores = []

    for index in range(input_ids.shape[1]):
        token_id = input_ids[0, index].item()
        # print(f'token_id: {token_id}')

        token_logits = logits[0, index - 1, :]
        # print(f"token_logits: {token_logits}")
        # print(f"token_logits.shape:{token_logits.shape}")

        # probability distribution over vocabulary
        token_probs = torch.softmax(token_logits, dim=-1)
        
        # print(f"token_probs: {token_probs}")
        # print(f"token_probs.shape: {token_probs.shape}")

        actual_token_prob = token_probs[token_id].item()
        # print(f"actual_token_prob: {actual_token_prob}")
        
        # Traditional surprisal of the actual token
        surprisal = -1 * np.log2(actual_token_prob)
        # print(f"surprisal: {surprisal}")
        # surprisal = -1 * math.log2(torch.nn.Softmax(0)(logits[0][index-1])[input_ids[0][index]].item())

        # print(f"surprisal_mit: {-1 * math.log2(torch.nn.Softmax(0)(logits[0][index-1])[input_ids[0][index]].item())}")

        surprisal_scores.append(surprisal)

        # KL Divergence term
        # Define Q(w_i), the smoothed distribution centered on the actual token
        q_dist = torch.full((vocab_size,), smooth_prob_mass / (vocab_size - 1),).to(device)
        # print(f"q_dist: {q_dist}")
        # print(f"q_dist.shape: {q_dist.shape}")
        q_dist[token_id] = 0.9


        # print(f"check sum ==1: {sum(q_dist)}")

        # # Calculate KL divergence: D_KL(P || Q)
        kl_divergence = torch.sum(token_probs * (torch.log(token_probs) - torch.log(q_dist))).item()
        # print(f"kl_divergence: {kl_divergence}")

        kl_term = gamma * kl_divergence
        cws = surprisal + kl_term

        cws_scores.append(cws)
        
    surprisal_per_token = list(zip(tokens,surprisal_scores[1:len(surprisal_scores)]))
    cws_per_token = list(zip(tokens, cws_scores[1:len(cws_scores)]))

    return surprisal_per_token, cws_per_token

def cws_t5(tokenizer, model, sentence, gamma=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    tokens = tokenizer.tokenize(sentence, return_tensors="pt")
        # print(tokens)

    input_ids = tokenizer(str(sentence), return_tensors="pt").input_ids.to(device)
    labels = tokenizer(str(sentence), return_tensors="pt").input_ids.to(device)

    logits = model(input_ids=input_ids, labels=labels).logits #Seq2SeqLMOutput

    # things for KL-Divergence
    vocab_size = logits.size(-1)
    smooth_prob_mass = 0.1
    
    surprisal_scores = []
    cws_scores = []

    for index in range(input_ids.shape[1]):
        token_id = input_ids[0, index].item()
        # print(f'token_id: {token_id}')

        token_logits = logits[0, index - 1, :]
        # print(f"token_logits: {token_logits}")
        # print(f"token_logits.shape:{token_logits.shape}")

        # probability distribution over vocabulary
        token_probs = torch.softmax(token_logits, dim=-1)
        
        # print(f"token_probs: {token_probs}")
        # print(f"token_probs.shape: {token_probs.shape}")

        actual_token_prob = token_probs[token_id].item()
        # print(f"actual_token_prob: {actual_token_prob}")
        
        # Traditional surprisal of the actual token
        surprisal = -1 * np.log2(actual_token_prob)
        # print(f"surprisal_cws: {surprisal}")

        # print(f"surprisal_mit: {-1 * math.log2(torch.nn.Softmax(0)(logits[0][index-1])[input_ids[0][index]].item())}")

        surprisal_scores.append(surprisal)

        # KL Divergence term
        # Define Q(w_i), the smoothed distribution centered on the actual token
        q_dist = torch.full((vocab_size,), smooth_prob_mass / (vocab_size - 1),).to(device)
        # print(f"q_dist: {q_dist}")
        # print(f"q_dist.shape: {q_dist.shape}")
        q_dist[token_id] = 0.9


        # print(f"check sum ==1: {sum(q_dist)}")

        # # Calculate KL divergence: D_KL(P || Q)
        kl_divergence = torch.sum(token_probs * (torch.log(token_probs) - torch.log(q_dist))).item()
        # print(f"kl_divergence: {kl_divergence}")

        kl_term = gamma * kl_divergence
        cws = surprisal + kl_term

        cws_scores.append(cws)
        
    surprisal_per_token = list(zip(tokens,surprisal_scores[1:len(surprisal_scores)]))
    cws_per_token = list(zip(tokens, cws_scores[1:len(cws_scores)]))

    return surprisal_per_token, cws_per_token


def get_next_token_probs_from_ids(input_ids: torch.Tensor, model) -> torch.Tensor:
    """
    Given a tensor of input IDs (shape: [1, seq_len]), compute the probability distribution
    over the next token using the model. The input_ids tensor is moved to the same device as the model.
    """

    # Move input_ids to the device of the model
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
        

    with torch.no_grad():
        if "flan-t5" in model.config._name_or_path:
            logits = model(input_ids=input_ids, labels=input_ids).logits

        else:
            outputs = model(input_ids)
            # logits shape: [batch_size, sequence_length, vocab_size]
            logits = outputs.logits

    # Get the logits for the next token (i.e. from the last position)
    next_token_logits = logits[0, -1, :]
    # Convert logits to probabilities
    probs = torch.softmax(next_token_logits, dim=-1)
    return probs


# def cis(tokenizer, model, context: str,):
#     """
#     Compute the Contextual Influence Score (CIS) for every token in the provided context.

#     For each token in the tokenized context, CIS is defined as the KL divergence between the
#     next-token distribution with the full context and the distribution after removing that token.

#     Returns:
#         A list of tuples (token_index, removed_token, cis_value)
#     """

#     # Determine the device of the model
#     device = next(model.parameters()).device

#     # Tokenize the full context (GPT-2 doesn't use special tokens by default)
#     full_ids = tokenizer.encode(context, add_special_tokens=False)

#     # Compute next-token distribution for the full context
#     full_tensor = torch.tensor([full_ids]).to(device)
#     full_probs = get_next_token_probs_from_ids(full_tensor, model)

#     results = []

#     # Loop over each token index in the tokenized context
#     for token_index in range(len(full_ids)):
#         # Create a reduced context by removing the token at token_index
#         reduced_ids = full_ids.copy()
#         removed_token_id = reduced_ids.pop(token_index)
#         reduced_tensor = torch.tensor([reduced_ids]).to(device)
#         reduced_probs = get_next_token_probs_from_ids(reduced_tensor, model)

#         # Compute the KL divergence between full_probs and reduced_probs:
#         # KL(P_full || P_reduced) = sum_i P_full(i) * (log(P_full(i)) - log(P_reduced(i)))
#         cis_value = torch.sum(full_probs * (torch.log(full_probs) - torch.log(reduced_probs))).item()

#         # Decode the removed token into a string
#         removed_token = tokenizer.decode([removed_token_id])
#         results.append((token_index, removed_token, cis_value))

#     return results


def cis(tokenizer, model, sentence: str,):
    """
    For each token x_i in the tokenized context (with 1 <= i <= len(context)-2, so that x_{i+1} exists),
    compute the influence of x_i on the probability of its immediate successor x_{i+1}.

    How current token (x_i) help/hurt to predict the next token (x_{i+1}).

    The influence is measured as the log ratio:
         log(P_full(x_{i+1}) / P_reduced(x_{i+1}))
    where:
         P_full is computed using the context [x_0, ..., x_i] (including x_i),
         P_reduced is computed using the context [x_0, ..., x_{i-1}] (without x_i).

    Returns:
         A list of tuples:
         (token_index, token_string, next_token_string, full_prob, reduced_prob, log_ratio)
    """
    full_ids = tokenizer.encode(sentence, add_special_tokens=False)
    print(f"full_ids:{full_ids}")
    all_info = []

    # Iterate from i = 1 to len(full_ids)-2 to ensure non-empty context and that x_{i+1} exists.
    for i in range(1, len(full_ids) - 1):
        # print(f"i:{i}")
        # Full context: [x_0, ..., x_i] to predict token after x_i (ideally x_{i+1})
        full_context_ids = full_ids[:i+1]
        full_tensor = torch.tensor([full_context_ids])
        full_probs = get_next_token_probs_from_ids(full_tensor, model)

        # Reduced context: [x_0, ..., x_{i-1}] (x_i removed) to predict token after x_{i-1}
        reduced_context_ids = full_ids[:i]
        reduced_tensor = torch.tensor([reduced_context_ids])
        reduced_probs = get_next_token_probs_from_ids(reduced_tensor, model)

        # The target token is x_{i+1}
        target_token_id = full_ids[i+1]

        # Get the probabilities of x_{i+1} under both contexts
        full_target_prob = full_probs[target_token_id]
        reduced_target_prob = reduced_probs[target_token_id]

        # Compute the log ratio as a measure of influence:
        # A positive log ratio indicates that including x_i increases the likelihood of x_{i+1}.
        log_ratio = torch.log(full_target_prob) - torch.log(reduced_target_prob)

        # Decode tokens for easier interpretation
        token_str = tokenizer.convert_ids_to_tokens(full_ids[i])
        next_token_str = tokenizer.convert_ids_to_tokens(target_token_id)

        all_info.append((i, token_str, next_token_str,
                        full_target_prob.item(), reduced_target_prob.item(), log_ratio.item()))

    log_ratios_only = [(x[1], float(x[-1])) for x in all_info]
    return all_info, log_ratios_only


def compute_entropy(prob_dist):
    """
    Compute the Shannon entropy given a probability distribution.
    Lower entropy means one token dominates the distribution.
    """

    # Avoid issues with log(0)
    nonzero_probs = prob_dist[prob_dist > 0]
    return -np.sum(nonzero_probs * np.log2(nonzero_probs))


# def eds(context: str, model, tokenizer):
#     """
#     Compute the Entropy-Based Dominance Score (EDS) for each token prediction in a sentence.

#     Parameters:
#         sentence (str): The input sentence.
#         model_name (str): Hugging Face model name (default: 'gpt2').

#     Returns:
#         List of tuples (token, entropy) for each token (except the final one).
#     """

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.eval()

#     input_ids = tokenizer.encode(context, return_tensors="pt").to(device)

#     # Get model outputs (logits) for each otken position.
#     # logits at position i are used to predict token at i+1
#     with torch.no_grad():
#         outputs = model(input_ids=input_ids)
#         logits = outputs.logits

#     # Convert logits to NumPy array for processing
#     logits_np = logits.cpu().numpy()[0]

#     # Calculate EDS for each token prediction.
#     # Compute the entory for each prediction position (except the last token)

#     eds_scores = []
#     for i in range(len(input_ids[0]) - 1):
#         # Get the logits for predicting th next token (position i+1)
#         token_logits = logits_np[i]

#         # Apply softmax to get the probability distribution
#         token_probs = softmax(token_logits)
#         print(f"token_probs: {token_probs}")
#         entropy = compute_entropy(token_probs)
#         print(f"entropy: {entropy}")
#         eds_scores.append(entropy)

#     print(f"eds_scores: {eds_scores}")
#     # Get token strings for reference
#     tokens = tokenizer.convert_ids_to_tokens(input_ids.cpu()[0])

#     # Pair each token (except the last) with its EDS score
#     results = list(zip(tokens[:-1], eds_scores))

#     return results


import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

def vanilla_entropy(tokenizer, model, sentence):
    """
    Compute the per-token entropy for a given sentence using a specified auto-regressive LM.
    
    Args:
        sentence (str): The input sentence.
        model: Hugging Face model.
    
    Returns:
        tokens (list): List of tokens (str) corresponding to the predicted tokens.
        entropies (list): List of entropy scores (float) for each token in nats.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Tokenize the sentence
    inputs = tokenizer(sentence, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        # Get model logits (shape: [batch, sequence_length, vocab_size])
        outputs = model(input_ids)
        logits = outputs.logits

        # For next-token prediction:
        #   - Remove the last logit (no prediction for the end-of-sequence)
        #   - Remove the first token in labels (no prediction for the first token)
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        # Convert logits to probabilities using softmax
        probs = F.softmax(shift_logits, dim=-1)

        # Compute token-level entropy: H = -sum(p * log(p)) for each token position
        token_entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

        # Convert the tensor to a list (squeeze batch dimension)
        token_entropy_list = token_entropy.squeeze().tolist()

        # Get tokens corresponding to the shifted labels (the predicted tokens)
        tokens = tokenizer.convert_ids_to_tokens(shift_labels.squeeze())

    return list(zip(tokens, token_entropy_list))

def token_convergence_depth():
    """Token Convergence Depth
    The number of layers the LLM take to converge to the right token.
    """
    pass


def surprisal_testing(text, tokenizer, model):
    """
    Compute surprisal values per token for a given text using the specified tokenizer and model.
    Returns a list of float values representing the surprisal of each token (excluding the first token).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # Tokenize text and build input_ids (including BOS token if available)
    # Note: Here we follow your logic, but ensure we return numerical values.
    tokens = tokenizer.tokenize(text, return_tensors="pt")
    
    if tokenizer.bos_token is not None:
        input_ids = torch.tensor(
            tokenizer.encode(tokenizer.bos_token) + tokenizer.encode(text)
        ).unsqueeze(0).to(device)
    else:
        input_ids = torch.tensor(
            tokenizer.encode(text)
        ).unsqueeze(0).to(device)
    
    # Get model logits and cast to float32
    logits = model(input_ids)[0].to(torch.float32)
    
    scores = []
    # Loop over token positions; note that index 0 would use logits[-1],
    # so we follow your original approach and then later discard the first score.
    for index in range(input_ids.shape[1]):
        # Apply softmax to logits for the previous token position.
        # (Using dim=0 as in your code; adjust if needed.)
        score = -1 * np.log2(
            torch.nn.Softmax(dim=0)(logits[0][index - 1])[input_ids[0][index]].item()
        )
        scores.append(score)
    
    # Return the scores excluding the first one (corresponding to the BOS token)
    return scores[1:]

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct").to(device)
    model.eval()

    print(model.config._name_or_path)

    # Example context
    # context = "It is a very useful behaviour when trying to get to the bottom of things and when you need to tease information out of the other person."
    context = "Blood and guys, adventure, that kind of thing."
    # Compute the CIS for each token in the context
    all_info, log_ratios_only = cis(tokenizer, model, context)


    # Display the results
    for index, token, next_token_str, full_target_prob, reduced_target_prob, cis_value in all_info:
        # print(f"Token index {index} ('{token}'): CIS = {cis_value:.5f}")
        print(f"Token index {index} ('{token}') predicts '{next_token_str}':")
        print(f"  Full context prob: {full_target_prob:.5f}")
        print(f"  Reduced context prob: {reduced_target_prob:.5f}")
        print(f"  CIS value: {cis_value:.5f}")

    print(f"-----")
    print(log_ratios_only)


    x = tokenizer(context).tokens()
    print(x)

    space_x = tokenizer(" "+context).tokens()
    print(space_x)
    # Compute the EDS for each token in the context
    # eds_results = eds(context, model, tokenizer)
    # print(eds_results)

    # tokens, entropies = vanilla_entropy(context, model, tokenizer)
    # for token, ent in zip(tokens, entropies):
    #     print(f"Token: {token}\tEntropy: {ent:.4f}")

    # x = list(zip(tokens, entropies))
    # print(x)

    tokenizer_yes = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    print(tokenizer_yes("Hi there my friend!").tokens())

    tokenizer_no = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

    print(tokenizer_no("Hi there my friend!", add_special_tokens=False).tokens())
    
    # tokenizer_qwen = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct-1M")
    # surprisal_values = surprisal(tokenizer, model, context)
    # print(f"surprisal_values: {surprisal_values}")

    # surprisal_testing_values = surprisal_testing(context, tokenizer, model)
    # print(f"surprisal_testing_values: {surprisal_testing_values}")
