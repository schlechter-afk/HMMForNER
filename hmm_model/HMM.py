import numpy as np

class HMM:
    def __init__(self, states, num_observations):
        self.states = states
        self.num_observations = num_observations
        self.start_probabilities = None
        self.transition_probabilities = None
        self.emission_probabilities = None

    def set_start_probabilities(self, start_probabilities):
        self.start_probabilities = start_probabilities

    def set_transition_probabilities(self, transition_probabilities):
        self.transition_probabilities = transition_probabilities

    def set_emission_probabilities(self, emission_probabilities):
        self.emission_probabilities = emission_probabilities

    def compute_start_probabilities(self, ner_tags):
        max_length = max(len(seq) for seq in ner_tags)
        ner_sequences_padded = [seq + [-1] * (max_length - len(seq)) for seq in ner_tags]
        ner_tags = np.array(ner_sequences_padded)
        start_states = ner_tags[:, 0]
        start_state_counts = np.bincount(start_states, minlength=len(self.states))
        self.start_probabilities = start_state_counts / len(self.states)

    def compute_transition_probabilities(self, ner_tags):
        self.transition_probabilities = np.zeros((len(self.states), len(self.states)))
        for sequence in ner_tags:
            for i in range(len(sequence) - 1):
                from_state = sequence[i]
                to_state = sequence[i + 1]
                self.transition_probabilities[from_state, to_state] += 1
        row_sums = self.transition_probabilities.sum(axis=1)
        self.transition_probabilities = self.transition_probabilities / row_sums[:, np.newaxis]

    def compute_emission_probabilities(self, ner_tags, observed_sequences):
        num_observations = len(set(obs for obs_seq in observed_sequences for obs in obs_seq))
        self.emission_probabilities = np.zeros((len(self.states), num_observations))
        observation_to_index = {obs: idx for idx, obs in enumerate(set(obs for obs_seq in observed_sequences for obs in obs_seq))}
        for i, sequence in enumerate(ner_tags):
            for j, state in enumerate(sequence):
                self.emission_probabilities[state, observation_to_index[observed_sequences[i][j]]] += 1
        row_sums = self.emission_probabilities.sum(axis=1)
        self.emission_probabilities = self.emission_probabilities / row_sums[:, np.newaxis]

    def viterbi_algorithm(self, obs):
        viterbi_table = [[0.0 for _ in range(len(self.states))] for _ in range(len(obs))]
        backpointer = [[0 for _ in range(len(self.states))] for _ in range(len(obs))]

        for t in range(len(obs)):
            for s in range(len(self.states)):
                if t == 0:
                    viterbi_table[t][s] = self.start_probabilities[s] * self.emission_probabilities[s][obs[t]]
                else:
                    max_prob = -1
                    max_backpointer = -1

                    for s_prime in range(len(self.states)):
                        prob = viterbi_table[t-1][s_prime] * self.transition_probabilities[s_prime][s] * self.emission_probabilities[s][obs[t]]
                        if prob > max_prob:
                            max_prob = prob
                            max_backpointer = s_prime

                    viterbi_table[t][s] = max_prob
                    backpointer[t][s] = max_backpointer

        best_path_pointer = max(range(len(self.states)), key=lambda s: viterbi_table[-1][s])
        best_path = [best_path_pointer]
        for t in range(len(obs)-1, 0, -1):
            best_path.insert(0, backpointer[t][best_path[0]])

        return best_path
    
    def train_hmm(self, sequences):
        self.compute_start_probabilities(sequences)
        self.compute_transition_probabilities(sequences)
        self.compute_emission_probabilities(sequences)

        