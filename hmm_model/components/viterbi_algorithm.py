import numpy as np
from start_probabilities import find_start_probabilities
from emission_probabilities import find_emission_probabilities
from transition_probabilities import find_transition_probabilities

def viterbi(observations, states, start_probabilities, transition_probabilities, emission_probabilities):
    num_states = len(states)
    num_observations = len(observations)
    
    viterbi_matrix = np.zeros((num_states, num_observations)) # Initializing the Viterbi matrix and backpointer matrix
    backpointer = np.zeros((num_states, num_observations), dtype=int)
    
    viterbi_matrix[:, 0] = start_probabilities * emission_probabilities[:, observations[0]]
    
    for t in range(1, num_observations):
        for s in range(num_states):
            max_prob = -1
            max_backpointer = -1
            
            # Computing the maximum probability and corresponding backpointer
            for s_prime in range(num_states):
                prob = viterbi_matrix[s_prime, t-1] * transition_probabilities[s_prime, s] * emission_probabilities[s, observations[t]]
                if prob > max_prob:
                    max_prob = prob
                    max_backpointer = s_prime
            
            viterbi_matrix[s, t] = max_prob
            backpointer[s, t] = max_backpointer
    
    best_path = [-1] * num_observations # We backtrack to find the best state sequence
    best_last_state = np.argmax(viterbi_matrix[:, num_observations - 1])
    best_path[-1] = best_last_state
    
    for t in range(num_observations - 2, -1, -1):
        best_last_state = backpointer[best_last_state, t + 1]
        best_path[t] = best_last_state

    best_path_tags = [states[i] for i in best_path] # Convert the best_path indices to actual NER tags

    return best_path_tags

# states = ['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'O']
# observations = [0, 1, 2, 3, 4]  # Index mapping of observations
# start_probabilities = np.array([0.1, 0.0, 0.3, 0.0, 0.6])
# transition_probabilities = np.array([[0.7, 0.1, 0.1, 0.0, 0.1],
#                                     [0.0, 0.7, 0.1, 0.1, 0.1],
#                                     [0.2, 0.1, 0.6, 0.0, 0.1],
#                                     [0.0, 0.2, 0.0, 0.7, 0.1],
#                                     [0.3, 0.0, 0.3, 0.0, 0.4]])
# emission_probabilities = np.array([[0.7, 0.0, 0.2, 0.0, 0.1],
#                                   [0.0, 0.8, 0.0, 0.1, 0.1],
#                                   [0.1, 0.0, 0.7, 0.1, 0.1],
#                                   [0.0, 0.1, 0.0, 0.8, 0.1],
#                                   [0.2, 0.0, 0.2, 0.0, 0.6]])

# result = viterbi(observations, states, start_probabilities, transition_probabilities, emission_probabilities)
# print(result)  # This will give you the best NER tags for the given observations.

sentences = [
    ["Apple", "Inc.", "is", "based", "in", "Cupertino", ",", "California"],
    ["John", "lives", "in", "New", "York", "City", "."],
]
ner_tags = [
    [0, 0, 1, 1, 1, 2, 1, 2], 
    [3, 1, 1, 2, 2, 2, 1],
]

states = list(set(tag for tag_seq in ner_tags for tag in tag_seq))
observations = list(set(word for sentence in sentences for word in sentence))
observations.append("unknown") 
num_states = len(states)

start_probabilities = find_start_probabilities(num_states, ner_tags)
emission_probabilities = find_emission_probabilities(num_states, ner_tags, sentences)
transition_probabilities = find_transition_probabilities(num_states, ner_tags)


new_sentence = ["Apple", "is", "located", "in", "Cupertino", "near", "John"]
observations = [observations.index(word) if word in observations else observations.index("unknown") for word in new_sentence]
result = viterbi(observations, states, start_probabilities, transition_probabilities, emission_probabilities)

predicted_tags = [states[i] for i in result]

print("Predicted NER Tags for the new sentence:")
for word, tag in zip(new_sentence, predicted_tags):
    print(f"{word}: {tag}")