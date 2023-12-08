import numpy as np

def find_emission_probabilities(num_states, ner_tags, observed_sequences):
    num_observations = len(set(obs for obs_seq in observed_sequences for obs in obs_seq))
    print(set(obs for obs_seq in observed_sequences for obs in obs_seq))
    emission_probabilities = np.zeros((num_states, num_observations))
    observation_to_index = {obs: idx for idx, obs in enumerate(set(obs for obs_seq in observed_sequences for obs in obs_seq))}
    
    for i, sequence in enumerate(ner_tags):
        for j, state in enumerate(sequence):
            emission_probabilities[state, observation_to_index[observed_sequences[i][j]]] += 1

    row_sums = emission_probabilities.sum(axis=1)
    emission_probabilities = emission_probabilities / row_sums[:, np.newaxis]

    return emission_probabilities


# Function Testing
num_states = 4
ner_tags = [
    [0, 1, 2, 0, 1],
    [2, 1, 0],
    [1, 2, 3, 0, 2],
    [0, 3],
]

observed_sequences = [
    ["apple", "banana", "cherry", "apple", "banana"],
    ["cherry", "banana", "apple"],
    ["banana", "cherry", "dog", "apple", "banana"],
    ["apple", "dog"],
]

print(find_emission_probabilities(num_states, ner_tags, observed_sequences))