import os
import pandas as pd

main_folder = 'indic_languages'

combined_train_data = []
combined_test_data = []

for language in os.listdir(main_folder):
    language_folder = os.path.join(main_folder, language)

    if os.path.isdir(language_folder):
        train_file_path = os.path.join(language_folder, f'{language}_train_data.csv')
        test_file_path = os.path.join(language_folder, f'{language}_test.csv')

        train_data = pd.read_csv(train_file_path)
        test_data = pd.read_csv(test_file_path)

        combined_train_data.append(train_data)
        combined_test_data.append(test_data)
        print("Combined train and test data for ", language)

combined_train = pd.concat(combined_train_data, ignore_index=True)
combined_test = pd.concat(combined_test_data, ignore_index=True)

combined_train.to_csv('combined_train.csv', index=False)
combined_test.to_csv('combined_test.csv', index=False)

print("Combined train and test data saved successfully.")