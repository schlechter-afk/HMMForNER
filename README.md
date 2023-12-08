# Project Mavericks:
## Team Members:
- Swayam Agrawal: Roll No. 2021101068
- Mitansh Kayathwal: Roll No. 2021101026

# Project Description: 
### The project requires the development of a Hidden Markov Model (HMM) for Named Entity Recognition (NER) from scratch.

# Requirements:
1. Creating the HMM architecture, defining states, transitions, and emission probabilities, and implementing algorithms for training and inference.
2. The project should involve testing the NER system on multiple corpora with different characteristics, domains, and languages.
3. The project should include data preprocessing steps, such as tokenization, stemming, and the removal of irrelevant information, to prepare the text data for NER.
4. Ensure that the HMM-based NER system is designed to be language-independent, making it adaptable for various languages and domains.
5. The project should emphasize that the NER system is not domain-specific, highlighting its versatility for recognizing named entities across different subject areas.

Training Datasets can be found:
- [here](https://huggingface.co/datasets/mitanshk/NER_HMM_Datasets)
- [here](https://huggingface.co/datasets/schlechter/NER_Datasets_SMAI/tree/main)

# Implementation:

1. <b>Data Preprocessing:</b> The first task is to perform preprocessing on the corpus. so as to make it suitable to be used in the HMM framework for all the languages. The training data may be collected from any source like from open source, tourism corpus or simply a plaintext file containing some sentences. Our preprocessing steps include:
    1)  Tokenization: Breaking down large blocks of text such as paragraphs and sentences into smaller, more manageable units.
    2) Stop/Noisy word removal: Stopwords refer to the most commonly occurring words in any natural language. These words do not add much value to the dataset, hence we remove them.
    3) Stemming: Reducing words to their base or root form. This helps in treating different inflections or derivations of a word as the same, improving the efficiency and accuracy of text analysis.
   
   The code for the same can be found in any of the language subfolders in the indic_languages folder of the repository. The name of the file would be `{language_name}_data_preprocessing.ipynb`.

2. <b>Creation of HMM Model Architecture:</b> We implemented a HMM class with methods to compute the start, transition and emission probabilities of the model. We also implemented the Viterbi algorithm for prediction of the named entities. The code for the same can be found in the `hmm_model/HMM.py` file in the repository.
3. <b>Training and Parameter estimation:</b> We trained the model on various training datasets and estimated the parameters of our HMM model. For training purposes we trained on:
   1) Indic Languages datasets: Hindi, Gujarati, Tamil, Telugu, Assamese. The code for the same can be found in the `hmm_model/HMMModel1.ipynb` and `hmm_model/HMMModel2.ipynb` files in the repository.
   2) Cross Domain datasets: AI, Literature, Science, Music, Political Corpus. The code for the preprocessing of these datasets and the datasets itself can be found in `cross_domain/{domain}` folder in the repository.

   - Note that there are two HMM models in the repository. Both these models differ in their handling of unseen data (i.e. there is a possibility that a word is observed in testing dataset but it is not present in the training dataset). The model will throw an error if we do not do this handling as the parameters corresponding to this unseen observation is not present (key-error).
  
  <strong>Handling Unseen Data:</strong>
  
   - <u>Method followed by first model (`hmm_model/HMMModel1.ipynb`) - <b>Using a pseudo-word “UNK”:</b></u> We insert a Pseudo-word while training the model. All words having a frequency less than a particular threshold (5 in our case), were relabelled as “UNK” and then the emission probabilities were computed. Thus, any new word in test dataset could also be mapped to this pseudoword for testing to ensure finite emission probability. This performs a closing effect on our Vocabulary. This is a common NLP technique for handling out of vocabulary words. (Courtesy:Reference Link(Columbia))
   - <u>Method followed by second model (`hmm_model/HMMModel2.ipynb`) - <b>Assigning a finite emission probability (`1e-4`)</b></u> to every out of vocabulary word. This ensures a finite emission probability so that we could perform the Viterbi Algorithm. This ensures a smoothing effect to the Emission probabilities without any contradictory change in the Emission Probability.

4. <b>Testing:</b> We tested our model on various test datasets. For testing purposes we tested on:
   1) Indic Languages datasets: Hindi, Gujarati, Tamil, Telugu, Assamese. The code for the same can be found in the `hmm_model/HMMModel1.ipynb` and `hmm_model/HMMModel2.ipynb` files in the repository.
   2) Cross Domain datasets: AI, Literature, Science, Music, Political Corpus. The code for the same can be found in `cross_domain/{domain}` folder in the repository.

    We also trained and tested the model on a combination of the above indic language training and testing datasets to create a multilingual training and testing dataset. We establish a correlation from the results obtained in the case of training/testing on a single language dataset and the multilingual dataset. The analysis and results can be found in the presentation attached in the repository.

    <strong><i> Evaluation metric: </i></strong> We evaluate the model's performance on its F1 Score.
    Why not accuracy?
    → Imbalanced Classes: In NER tasks, entities of interest (e.g., person names, locations) are often a small proportion of the overall text. If one class ("Others") significantly outweighs the others, a model that simply predicts the majority class for every instance could achieve a high accuracy but fail to identify the entities, which is the primary goal of NER. 
    The F1-score is less affected by class imbalance and provides a more comprehensive evaluation.

5. <b>Results:</b> For results and inference analysis refer to presentation ppt attached in the repository. The results prove that the HMM-based NER system is designed to be language-independent and that it is not domain-specific, highlighting its versatility for recognizing named entities across different subject areas.
6. <b>Comparison with SOTA models:</b> To compare our model with better models utilizing Deep Learning Methods like Bidirectional-LSTM-CNNs (SOTA), we evaluated our HMM Model on CoNLL-2003 dataset, the weighted average F1 score for the SOTA model is `90.9%` with ~70 epochs. We obtained a weighted average F1 score of `64%`. The code for the same can be found in the `hmm_model/lstm_comparison.ipynb` file in the repository. The dataset (CONLL-2003) can be found in the `cross_domain/lstm_data` folder in the repository. The preprocessing steps for the same can be found in the `cross_domain/LSTMPaperData_preprocessing.ipynb` file in the repository.
7. <b>Analysis of the reasons behind the lagging of HMM model in comparison to the SOTA Model: </b> 
- HMMs are inherently unidirectional, meaning they model sequences from left to right or right to left but not both simultaneously.
- Bidirectional LSTM-CNN models capture contextual information from both past and future words for each word in the sequence. Bidirectional models process the input sequence in both forward and backward directions, allowing them to capture dependencies from both sides. 
- LSTMs, with their ability to capture long-range dependencies, can effectively model relationships between words that are farther apart in the sequence. 
- Example: Consider the sentence: <i><u>"Apple Inc. is planning to open a new store in Paris."</u></i>
  In this sentence, a unidirectional HMM might struggle to accurately identify the entity boundaries. For example, it might have difficulty determining whether `"Apple"` is part of the entity `"Apple Inc."` or if `"Inc."` is a separate entity. A bidirectional LST-CNN model, on the other hand, could leverage information from both directions. It may better understand that `"Apple Inc."` is an organization and correctly identify the entities resulting in a better model performance.

8. <b> Concluding Remarks </b>: The paper suggests that on a rich corpora, their performance reaches till about 90%. However, after testing on multiple corpora we were not able to verify such kind of performance. The paper does not provide any proofs for it’s result. We however have verified that HMM as a model is generalizing very nicely across Cross Domain and Multilingual datasets. Our results thus lead us to conclude that HMM for NER is indeed Language independent and invariant to Cross Domain or for that matter any corpus which is fed to it. However, we cannot ignore the training data dependence in order for HMMs to work well for any corpora. Thus, we need to maintain caution while evaluating our HMM Model and drawing inferences from it. For a rich corpora, the model does tend to perform well and generalize better.

9. <b> References: </b> @misc{author = {Mhaske, Arnav and Kedia, Harshit and Doddapaneni, Sumanth and Khapra, Mitesh M. and Kumar, Pratyush and Murthy, Rudra and Kunchukuttan, Anoop}, title = {Naamapadam: A Large-Scale Named Entity Annotated Data for Indic Languages}}

https://github.com/kamalkraj/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs/tree/master : SOTA model used for comparison with our HMM Model.

https://arxiv.org/pdf/2012.04373.pdf : SOTA model for Cross Domain NER.



This marks the end of the project by completing all the requirements mentioned in the project description.

Regards,
Team Mavericks.
