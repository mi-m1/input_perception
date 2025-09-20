import statistics


############# finder functions #####################
def find_subsequence_positions(tuples_list, tokens):
    # Extract only the token strings (ignoring values).
    all_tokens = [t[0] for t in tuples_list]
    
    # Length of the short sequence
    seq_len = len(tokens)
    
    # Search using a simple sliding window
    for i in range(len(all_tokens) - seq_len + 1):
        # Compare slice of all_tokens to our short sequence
        if all_tokens[i : i + seq_len] == tokens:
            return i, i + seq_len - 1
    
    # If we reach here, the subsequence wasn't found
    return None, None

def get_list_of_values_for_phrase(values_for_phrase, positions_phrase):
    '''
    Get the list of values for the tokens in the phrase.
    Use cws_values or surprisal_values as the values_for_phrase, i.e., [(token, value)]
    '''
    start, end = positions_phrase
    return [float(t[1]) for t in values_for_phrase[start:end+1]]

def take_context(my_list, index_tuple):
    start = index_tuple[0]
    end = index_tuple[1]
    return my_list[:start] + my_list[end+1:]

def safe_stdev(data):
    if len(data) > 1:
        return statistics.stdev(data)
    else:
        return 0  
    
def safe_division(numerator, denominator, default=0):
    """Returns numerator / denominator, avoiding division by zero by returning a default value."""
    return numerator / denominator if denominator != 0 else default

# def find_middle(lst):
#     '''
#     Find the middle element of a list. If the list is even, return the element to the left of the middle.'''
#     if not lst:  # Check if the list is empty
#         return None
#     mid = len(lst) // 2
#     return lst[mid - 1] if len(lst) % 2 == 0 else lst[mid]

def find_middle(lst):
    '''
    Find the middle element of a list. If the list is even, return the element to the right of the middle.'''
    if not lst:  # Check if list is empty
        return None
    mid = len(lst) // 2  # Always picks the right middle if even
    return lst[mid]

def extract_values_from_list_of_tuple(data):
    """
    Extracts numerical values from a list of tuples.
    
    Parameters:
    data (list of tuples): A list where each tuple contains a string and a numerical string.

    Returns:
    list of floats: Extracted numerical values as floats.
    """
    return [float(value) for _, value in data]


def find_tuple_positions(tuples_list, tokens):
    """
    Returns (start_index, end_index) such that:
      tuples_list[start_index][0] == tokens[0]
      tuples_list[end_index][0] == tokens[-1]
    """
    if not tokens:
        return None, None  # or raise an error
    
    first_token = tokens[0]
    last_token  = tokens[-1]
    
    start_index = None
    end_index   = None
    
    # Find the first occurrence of the first token
    for i, (text, value) in enumerate(tuples_list):
        if text == first_token:
            start_index = i
            break  # stop as soon as we find it
    
    # If we never found the first token, nothing else to do
    if start_index is None:
        return None, None
        
    for i in range(start_index, len(tuples_list)):
        if tuples_list[i][0] == last_token:
            end_index = i
            break
    
    return start_index, end_index-1


##################### feature calculation fuctions ######################

# class for a metric
# big function that takes in all lists for each metric and calculates all the metrics

