import sys
import pandas as pd
import gensim.downloader as api
import csv
import numpy as np
import random

# Declare and initialize variables
df = pd.read_csv('synonyms.csv')
question_list = df['question'].values.tolist()
answers_list = df['answer'].values.tolist()
word_list = []
for i in range(4):
    word_list.append(df[str(i)].values.tolist())


# Compares the similarity using the provided model
def synonym_test_dataset(model, file_name):
    f = open(file_name + '-details.csv', 'w')
    results = []
    # create the csv writer
    writer = csv.writer(f)

    for i in range(len(question_list)):
        # initialize variables
        sim_value = -sys.maxsize - 1
        label = 'wrong'
        if file_name == 'baseline':
            chosen_word = word_list[random.randint(0,3)][i]
        else:
            for j in range(4):
                try:
                    approximation = model.similarity(question_list[i], word_list[j][i])
                    if approximation > sim_value:
                        sim_value = approximation
                        chosen_word = word_list[j][i]
                except:
                    chosen_word = word_list[j][i]
                    label = 'guess'
                    break
        # check if model choice is correct
        if chosen_word == answers_list[i] and (model == 'baseline' or label != 'guess'):
            label = 'correct'

        writer.writerow([question_list[i], answers_list[i], chosen_word, label])
        results.append(tuple([question_list[i], answers_list[i], chosen_word, label]))
    f.close()
    return results


def analysis(results, model_name):
    # create csv file
    f = open('analysis.csv', 'a')
    # create the csv writer
    results = np.array(results)
    correct_count = np.count_nonzero(results[:, 3] == 'correct')
    # (80 - number of guesses)
    non_guess_count = len(question_list) - \
        np.count_nonzero(results[:, 3] == 'guess')
    # api info

    number_of_unique_words = api.info(model_name)['num_records'] if model_name != 'baseline' else 'N/A'

    try:
        accuracy = correct_count / non_guess_count
    except:
        accuracy = 'invalid'

    writer = csv.writer(f)
    writer.writerow([model_name, number_of_unique_words,
                     correct_count, non_guess_count, accuracy])
    f.close()

###################################################################################


# TASK 1
# ---- Part 1 ----
model_name = 'word2vec-google-news-300'
model = api.load(model_name)
results = synonym_test_dataset(model, model_name)

# ---- Part 2 ----
analysis(results, model_name)

###################################################################################

# TASK 2

# random baseline

model_name = 'baseline'
results = synonym_test_dataset(None, model_name)
analysis(results, model_name)

# model 1

model_name = 'glove-wiki-gigaword-300'
model = api.load(model_name)
results = synonym_test_dataset(model, model_name)
analysis(results, model_name)

# model 2

model_name = 'fasttext-wiki-news-subwords-300'
model = api.load(model_name)
results = synonym_test_dataset(model, model_name)
analysis(results, model_name)

# model 3

model_name = 'glove-twitter-25'
model = api.load(model_name)
results = synonym_test_dataset(model, model_name)
analysis(results, model_name)

# model 4

model_name = 'glove-twitter-100'
model = api.load(model_name)
results = synonym_test_dataset(model, model_name)
analysis(results, model_name)

###################################################################################
