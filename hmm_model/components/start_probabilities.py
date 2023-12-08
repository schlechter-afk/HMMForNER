import numpy as np

def find_start_probabilities(num_states, ner_tags):
    max_length = max(len(seq) for seq in ner_tags)

    ner_sequences_padded = [seq + [-1] * (max_length - len(seq)) for seq in ner_tags]

    ner_tags = np.array(ner_sequences_padded)

    start_states = ner_tags[:, 0]
    start_state_counts = np.bincount(start_states, minlength=num_states)
    
    start_probabilities = start_state_counts / num_states

    return start_probabilities

# Function Testing
num_states = 4
ner_tags = [
    [2,3,1],
    [1,2,3,0],
    [3,1,2,0],
    [0,1]
    ]


# print(find_start_probabilities(num_states, ner_tags))

