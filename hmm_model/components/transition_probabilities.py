import numpy as np

def find_transition_probabilities(num_states, ner_tags):
    transition_probabilities = np.zeros((num_states, num_states))
    for sequence in ner_tags:
        for i in range(len(sequence) - 1):
            from_state = sequence[i]
            to_state = sequence[i + 1]
            transition_probabilities[from_state, to_state] += 1

    row_sums = transition_probabilities.sum(axis=1)
    transition_probabilities = transition_probabilities / row_sums[:, np.newaxis]

    return transition_probabilities

# Function Testing
num_states = 4
ner_tags = [
    [2,3,1],
    [1,2,3,0],
    [3,1,2,0],
    [0,1]
    ]


# print(find_transition_probabilities(num_states, ner_tags))