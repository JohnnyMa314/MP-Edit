import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
from simpletransformers.classification import ClassificationModel
import random
import spacy
import sys
from spacy.lang.en.stop_words import STOP_WORDS
import pandas as pd
from collections import Counter
from bert_score import BERTScorer
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForMaskedLM
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from tqdm import tqdm
from scipy.special import softmax
import string
from fairseq.models.bart import BARTModel
import datetime
import gc
from fairseq.data.data_utils import collate_tokens

def train_NLI(train_df, n_epoch, out_dir):
    args = {
        'num_train_epochs': n_epoch,
        'overwrite_output_dir': True,
        'learning_rate': 1e-5
    }
    if torch.cuda.is_available():
        model = ClassificationModel("roberta", "roberta-base", use_cuda=True, args=args, num_labels=3)
    else:
        model = ClassificationModel("roberta", "roberta-base", use_cuda=False, args=args, num_labels=3)

    model.train_model(train_df, output_dir = out_dir)

    return model

def gen_MP_Edits_IMDB(df, num_reviews, fill_models, n_grams, nlp, bart, scorer):
    # predefine dataframe
    data = pd.DataFrame(columns=['review-num', 'sentence-num', 'pred-model', 'fill-model',
                                 'original', 'mask-filled', 'token_changes', 'n_gram', 'Word2Vec-Score', 'Bert-Score',
                                 'gold-label'])  # setup

    # for each review, get sentences, mask content words, replace mask with Mask-Fill top predictions
    for index, row in tqdm(df[0:int(num_lines)].iterrows(), total=df[0:int(num_lines)].shape[0]):
        text = row['text']
        sentences = nlp(text).sents
        j = 0
        for sentence in sentences:
            sentence = nlp(sentence.text)
            if 5 < len(sentence) < 30:  # if sentence has more than 5 and less than 30 tokens.
                for token in sentence:
                    if token.pos_ == 'NOUN' or token.pos_ == 'ADJ' or token.pos_ == 'VERB' and token.text != '':  # spacy POS for content
                        for N in n_grams:
                            ngrams = [sentence[i:i + N] for i in
                                      range(token.i - N + 1, token.i + 1)]  # get ngrams around content token
                            for ngram in ngrams:
                                if 'BART' in fill_models and ngram.text != '':
                                    masked_sent = sentence.text.replace(ngram.text, '<mask>')  # mask each ngram
                                    candidates = bart.fill_mask(masked_sent, topk=10, beam=10, match_source_len=False)

                                    # evaluating fill in candidates. add more filters later.
                                    for sent, prob in candidates:
                                        new_token = ' '.join(
                                            list((Counter(sent.split(' ')) - Counter(
                                                sent.split(' '))).keys()))  # get new words
                                        if sentence.text != sent and new_token not in STOP_WORDS:
                                            filled_sent = sent
                                            token_prob = prob
                                            break

                                    new_token = ' '.join(list((Counter(filled_sent.split(' ')) - Counter(
                                        masked_sent.split(' '))).keys()))  # get new words
                                    data = data.append({'review-num': index,
                                                        'sentence-num': j,
                                                        'pred-model': 'imdb-BaseBert',
                                                        'fill-model': 'BaseBart',
                                                        'original': sentence.text,
                                                        'mask-filled': filled_sent,
                                                        'token_changes': (ngram.text, new_token),
                                                        'fill_prob': token_prob.item(),
                                                        'n_gram': N,
                                                        'Word2Vec-Score': nlp(sentence.text).similarity(
                                                            nlp(filled_sent)),
                                                        'Bert-Score': scorer.score([sentence.text], [filled_sent])[
                                                            2].item(),
                                                        'gold-label': row.label}, ignore_index=True)
            j += 1
        data.to_csv('output/' + str(num_reviews) + '_imdb.csv')

    return data

def gen_MP_Edits_NLI(df, num_lines, fill_models, n_grams, nlp, bart, scorer, filename):
    if 'mnli' in filename:
        pred_model = 'mnli-roberta'
    if 'snli' in filename:
        pred_model = 'snli-roberta'
    # predefine dataframe
    prems = pd.DataFrame(columns=['line-num', 'pred-model', 'fill-model',
                                 'premise', 'hypothesis', 'mask-filled', 'token_changes', 'fill_prob', 'n_gram', 'Word2Vec-Score',
                                 'Bert-Score', 'gold-label'])  # setup

    hypos = pd.DataFrame(columns=['line-num', 'pred-model', 'fill-model',
                                 'premise', 'hypothesis', 'mask-filled', 'token_changes', 'fill_prob', 'n_gram', 'Word2Vec-Score',
                                 'Bert-Score', 'gold-label'])

    # for each review, get sentences, mask content words, replace mask with Mask-Fill top predictions
    for index, row in tqdm(df[0:int(num_lines)].iterrows(), total=df[0:int(num_lines)].shape[0]):
        # change premise
        text = row['text_a']
        sentence = nlp(text)
        if 5 < len(sentence) < 30:  # if sentence has more than 5 and less than 30 tokens.
            for token in sentence:
                if token.pos_ == 'NOUN' or token.pos_ == 'ADJ' or token.pos_ == 'VERB' and token.text != '':  # spacy POS for content
                    for N in n_grams:
                        ngrams = [sentence[i:i + N] for i in range(token.i - N + 1, token.i + 1)]  # get ngrams around content token
                        for ngram in ngrams:
                            if 'BART' in fill_models and ngram.text != '':
                                masked_sent = sentence.text.replace(ngram.text, '<mask>')  # mask each ngram
                                candidates = [(sent, prob) for sent, prob in
                                              bart.fill_mask(masked_sent, topk=3, beam=10, match_source_len=False)]
                                print(candidates)
                                print(candidates[::-1])
                                # evaluating fill in candidates. add more filters later.
                                for sent, prob in candidates[::-1]:
                                    new_token = ' '.join(
                                        list((Counter(sent.split(' ')) - Counter(
                                            sent.split(' '))).keys()))  # get new words
                                    if sentence.text != sent and new_token not in STOP_WORDS:
                                        filled_sent = sent
                                        token_prob = prob
                                        break

                                new_token = ' '.join(list((Counter(filled_sent.split(' ')) - Counter(
                                    masked_sent.split(' '))).keys()))  # get new words
                                prems = prems.append({'line-num': index,
                                                    'pred-model': pred_model,
                                                    'fill-model': 'BaseBart',
                                                    'premise': sentence.text,
                                                    'hypothesis': row['text_b'],
                                                    'mask-filled': filled_sent,
                                                    'token_changes': (ngram.text, new_token),
                                                    'fill_prob': token_prob.item(),
                                                    'n_gram': N,
                                                    'Word2Vec-Score': nlp(sentence.text).similarity(nlp(filled_sent)),
                                                    'token-similarity': nlp(ngram.text).similarity(nlp(new_token)),
                                                    'Bert-Score': scorer.score([sentence.text], [filled_sent])[2].item(),
                                                    'gold-label': row.labels}, ignore_index=True)

        # change hypothesis
        text = row['text_b']
        sentence = nlp(text)
        if 5 < len(sentence) < 30:  # if sentence has more than 5 and less than 30 tokens.
            for token in sentence:
                if token.pos_ == 'NOUN' or token.pos_ == 'ADJ' or token.pos_ == 'VERB' and token.text != '':  # spacy POS for content
                    for N in n_grams:
                        ngrams = [sentence[i:i + N] for i in
                                  range(token.i - N + 1, token.i + 1)]  # get ngrams around content token
                        for ngram in ngrams:
                            if 'BART' in fill_models and ngram.text != '':
                                masked_sent = sentence.text.replace(ngram.text, '<mask>')  # mask each ngram
                                candidates = [(sent, prob) for sent, prob in
                                              bart.fill_mask(masked_sent, topk=3, beam=10, match_source_len=False)]
                                print(candidates)
                                print(candidates[::-1])
                                # evaluating fill in candidates. add more filters later.
                                for sent, prob in candidates[::-1]:
                                    new_token = ' '.join(
                                        list((Counter(sent.split(' ')) - Counter(
                                            sent.split(' '))).keys()))  # get new words
                                    if sentence.text != sent and new_token not in STOP_WORDS:
                                        filled_sent = sent
                                        token_prob = prob
                                        break

                                new_token = ' '.join(list((Counter(filled_sent.split(' ')) - Counter(
                                    masked_sent.split(' '))).keys()))  # get new words
                                hypos = hypos.append({'line-num': index,
                                                    'pred-model': pred_model,
                                                    'fill-model': 'BaseBart',
                                                    'premise': row['text_a'],
                                                    'hypothesis': sentence.text,
                                                    'mask-filled': filled_sent,
                                                    'token_changes': (ngram.text, new_token),
                                                    'fill_prob': token_prob.item(),
                                                    'n_gram': N,
                                                    'Word2Vec-Score': nlp(sentence.text).similarity(nlp(filled_sent)),
                                                    'token-similarity': nlp(ngram.text).similarity(nlp(new_token)),
                                                    'Bert-Score': scorer.score([sentence.text], [filled_sent])[2].item(),
                                                    'gold-label': row.labels}, ignore_index=True)

        hypos.to_csv(filename + '_hypos.csv')
        prems.to_csv(filename + '_prems.csv')

    return hypos, prems

def predict_on_MP_imdb(df, imdb_model, num_lines):
    # filling in model predictions for each sentence
    orgi_sents = np.array(df['original'])
    mask_sents = np.array(df['mask-filled'])

    # generating predictions from fine tuned model
    orig_preds, orig_raw_outputs = imdb_model.predict(orgi_sents)
    mask_preds, mask_raw_outputs = imdb_model.predict(mask_sents)

    # probabilities of classes
    orig_prob = softmax(orig_raw_outputs, axis=1)
    mask_prob = softmax(mask_raw_outputs, axis=1)

    # probability of original class
    mask_probs = []
    for i in range(len(mask_preds)):
        mask_probs.append(mask_prob[i, orig_preds[i]])

    orig_probs = []
    for i in range(len(orig_preds)):
        orig_probs.append(orig_prob[i, orig_preds[i]])

    # putting into data
    df['orig-label'] = orig_preds
    df['new-label'] = mask_preds
    df['orig-label-prob'] = [round(float(i), 2) for i in orig_probs]
    df['new-label-prob'] = [round(float(i), 2) for i in mask_probs]

    #print('changed labels: ' + "{0:.00%}".format((sum(orig_preds != mask_preds) / len(mask_preds))))  # 7%

    # changes to labels
    df['label-changed'] = (df['orig-label'] != df['new-label'])
    df['label-prob-diff'] = np.abs(df['orig-label-prob'] - df['new-label-prob'])

    # TEMPORARY SOLUTION TO BAD BART INFILLLINGS
    df = df[df['Word2Vec-Score'] > 0.8]

    return df

# predicting 
def predict_on_MP_NLI(df, NLI_model, num_lines, changed_sentence):
    # filling in model predictions and probabilities for each sentence
    mask_sent = df['mask-filled']
    masked_pairs = []
    orig_pairs = []

    if changed_sentence == "premise":
        orig_sent = df['premise']
        unchanged = df['hypothesis']
        for i in range(len(mask_sent)):
            masked_pairs.append([mask_sent[i], unchanged[i]])
            orig_pairs.append([orig_sent[i], unchanged[i]])
        
    if changed_sentence == "hypothesis":
        orig_sent = df['hypothesis']
        unchanged = df['premise']
        for i in range(len(mask_sent)):
            masked_pairs.append([unchanged[i], mask_sent[i]])
            orig_pairs.append([unchanged[i], orig_sent[i]])
    if changed_sentence != 'hypothesis' and changed_sentence != 'premise':
        print("Please chose premise or hypothesis for changed sentence.")

    # generating batch predictions from fine tuned model
    mask_batch = collate_tokens([NLI_model.encode(pair[0], pair[1]) for pair in masked_pairs], pad_idx=1)
    orig_batch = collate_tokens([NLI_model.encode(pair[0], pair[1]) for pair in orig_pairs], pad_idx=1)

    # cut into smaller batches to store on GPU mem
    batch_size = 16
    mask_batches = [mask_batch[i * batch_size:(i + 1) * batch_size] for i in range((len(mask_batch) + batch_size - 1) // batch_size )]
    orig_batches = [orig_batch[i * batch_size:(i + 1) * batch_size] for i in range((len(orig_batch) + batch_size - 1) // batch_size )]  

    # use lists for speed
    mask_logprobs = []
    orig_logprobs = []
    mask_logprobs = torch.Tensor().cuda()
    orig_logprobs = torch.Tensor().cuda()

    # generate predictions in batches
    torch.cuda.empty_cache()
    with torch.no_grad():
        for i in tqdm(range(len(mask_batches))):
            mbatch = mask_batches[i].cuda()
            obatch = orig_batches[i].cuda()
            mask_logprobs = torch.cat((mask_logprobs, NLI_model.predict('mnli', mbatch)), 0)
            orig_logprobs = torch.cat((orig_logprobs, NLI_model.predict('mnli', obatch)), 0)
            
            # required for memory
            del mbatch
            del obatch
            gc.collect()
            
    # list of tensors into tensor list

    mask_preds = mask_logprobs.argmax(axis=1).to('cpu')
    orig_preds = orig_logprobs.argmax(axis=1).to('cpu')

    mask_logprobs = mask_logprobs.to('cpu')
    orig_logprobs = orig_logprobs.to('cpu')

    # filling out probabilities of original class.
    same_probs = []
    for i in range(len(mask_preds)):
        same_probs.append(round(float(np.exp(mask_logprobs[i,[orig_preds[i].item()]]).item()), 2))

    orig_probs = []
    for i in range(len(orig_preds)):
        orig_probs.append(round(float(np.exp(orig_logprobs[i,[orig_preds[i].item()]]).item()), 2))

    mask_probs = []
    for i in range(len(orig_preds)):
        mask_probs.append(round(float(np.exp(mask_logprobs[i,[mask_preds[i].item()]]).item()), 2))

    # putting into data
    df['orig-label'] = orig_preds
    df['new-label'] = mask_preds
    df['orig-label-prob'] = orig_probs
    df['same-label-prob'] = same_probs
    df['new-label-prob'] = mask_probs

    df['label-changed'] = df['orig-label'] != df['new-label']

    # labels. fix later.
    df.loc[df['orig-label'] == 0, ['orig-label']] = 'contradiction'
    df.loc[df['orig-label'] == 1, ['orig-label']] = 'neutral'
    df.loc[df['orig-label'] == 2, ['orig-label']] = 'entailment'

    df.loc[df['new-label'] == 0, ['new-label']] = 'contradiction'
    df.loc[df['new-label'] == 1, ['new-label']] = 'neutral'
    df.loc[df['new-label'] == 2, ['new-label']] = 'entailment'

    print('changed: ' + "{0:.00%}".format(sum(orig_preds != mask_preds)/len(mask_preds)))

    df['same-label-prob-diff'] = [np.abs(i-j) for i, j in zip(orig_probs, same_probs)]

    #TEMPORARY SOLUTION TO BUGGY BART INFILLING 
    df = df[df['Word2Vec-Score'] > 0.8]
    contrast_set = df[df['label-changed'] == True]

    return df, contrast_set

# computing metrics for a tagged data frame
def compute_metrics(tagged_df):
    # compute various metrics
    from ast import literal_eval as make_tuple
    from collections import Counter

    CS_data = tagged_df[tagged_df['label-changed'] == True]

    # word changes that shift label
    words = [make_tuple(w1) for w1 in CS_data['token_changes']]
    change_count = Counter(words)
    print(change_count)

    # count of masked words
    filled_words = [make_tuple(w1)[1] for w1 in CS_data['token_changes']]
    filled_count = Counter(filled_words)
    print(filled_count)

    # count of label switches in contrast set
    switches = zip(CS_data['orig-label'], CS_data['new-label'])
    switch_changes = Counter(switches)
    print(switch_changes)

# creating similarity plots
def similarity_figures(tagged_df, scoring_func):
    ### general plots
    plt.clf()

    # plot BERT Score for various slices
    x = tagged_df[tagged_df['label-changed'] != True][scoring_func]
    y = tagged_df[tagged_df['label-changed'] == True][scoring_func]
    a = tagged_df[tagged_df['fill-model'] == 'BaseBart'][scoring_func]
    b = tagged_df[tagged_df['fill-model'] == 'BaseBert'][scoring_func]
    bins = np.linspace(0.96, 1, 50)

    # compare same vs changed label similarity
    plt.hist(x, bins,
             weights=np.ones(len(tagged_df[tagged_df['label-changed'] != True])) / len(
                 tagged_df[tagged_df['label-changed'] != True]),
             alpha=0.5, label='SameLabel')
    plt.hist(y, bins,
             weights=np.ones(len(tagged_df[tagged_df['label-changed'] == True])) / len(
                 tagged_df[tagged_df['label-changed'] == True]),
             alpha=0.5, label='ChangedLabel')
    plt.legend(loc='upper left')
    plt.savefig('BERT Similarity by Label Change.png')
    plt.clf()

    # compare BART and BERT generated similarity
    plt.hist(a, bins, weights=np.ones(len(tagged_df[tagged_df['fill-model'] == 'LargeBart'])) / len(
        tagged_df[tagged_df['fill-model'] == 'LargeBart']),
             alpha=0.5, label='LargeBart')
    plt.hist(b, bins,
             weights=np.ones(len(tagged_df[tagged_df['fill-model'] == 'BaseBert'])) / len(
                 tagged_df[tagged_df['fill-model'] == 'BaseBert']),
             alpha=0.5, label='BaseBert')
    plt.legend(loc='upper left')
    plt.savefig('BERT Similarity by Fill in Model.png')
    plt.show()

# Parsing inputs from command line
def parse_inputs():
    process = True

    try:
        dataset = sys.argv[1]
    except:
        print("Enter the dataset (IMDB, MNLI, SNLI) as the first argument.")
        process = False

    try:
        num_lines = sys.argv[2]
    except:
        print("Enter the number of lines to generate over as the second argument. Make sure data is in correct folder.")
        process = False

    try:
        data_action = sys.argv[3]
    except:
        print("Please enter which action to take on this dataset.")
        process = False

    if process:
        return (dataset, num_lines, data_action)
    else:
        return False


def main(dataset, num_lines, data_action):
    random.seed(1996)

    print(dataset)
    print(num_lines)
    print(data_action)

    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
        print("Cuda is on")

    ## IMPORTING RELEVANT DATASETS
    if 'IMDB' in dataset:
        # processing IMDB data
        imdb = pd.read_csv('data/IMDB-Dataset.csv')
        imdb['label'] = (imdb['sentiment'] == 'positive').astype(int)
        imdb.rename({'review': 'text'}, axis=1, inplace=True)
        imdb.drop('sentiment', axis=1, inplace=True)
        imdb['text'] = [i.replace('<br /><br />', '') for i in imdb['text']]
        imdb = imdb.sample(frac=1)

    # processing MNLI data
    if 'MNLI' in dataset:
        mnli = pd.read_csv('data/multinli_1/multinli_1_train.txt', sep="\t", error_bad_lines = False)  # need to factor
        mnli['gold_label'].loc[mnli['gold_label'] == 'entailment'] = 0
        mnli['gold_label'].loc[mnli['gold_label'] == 'neutral'] = 1
        mnli['gold_label'].loc[mnli['gold_label'] == 'contradiction'] = 2
        mnli = mnli[mnli['gold_label'] != '-']
        mnli = mnli.sample(frac=1)

        mnli_train = mnli[['sentence1', 'sentence2', 'gold_label']]
        mnli_train.columns = ["text_a", "text_b", "labels"]

    # processing SNLI data
    if 'SNLI' in dataset:
        snli = pd.read_csv('data/snli_1/snli_1_test.txt', sep="\t")
        snli['gold_label'].loc[snli['gold_label'] == 'entailment'] = 0
        snli['gold_label'].loc[snli['gold_label'] == 'neutral'] = 1
        snli['gold_label'].loc[snli['gold_label'] == 'contradiction'] = 2
        snli = snli[snli['gold_label'] != '-']
        snli = snli.sample(frac=1)

        snli_train = snli[['sentence1', 'sentence2', 'gold_label']]
        snli_train.columns = ["text_a", "text_b", "labels"]

    ## TASK: FINETUNE PRE-TRAINED MODEL FOR SENTIMENT OR ENTAILMENT CLASSIFICATION.
    if data_action == 'Train':
        if 'IMDB' in dataset:
            imdb_model = train_IMDB(df, n_epoch= 3)
        if 'MNLI' in dataset:    
            MNLI_model = train_NLI(mnli_train, n_epoch = 3, out_dir = 'roberta-mnli')
        if 'SNLI' in dataset:
            SNLI_model = train_NLI(snli_train, n_epoch=3, out_dir = 'roberta-snli')

    ## TASK: MASK-INFILL CONTENT NGRAMS WITH BART MODEL.
    if data_action == 'Generate':
        # loading NLP packages
        nlp = spacy.load('en_core_web_lg')
        scorer = BERTScorer(lang="en")
        bart = BARTModel.from_pretrained('bart.base', checkpoint_file='model.pt')
        bart.eval()
        if torch.cuda.is_available():
            bart.cuda()

        # generating MP Edits
        if 'IMDB' in dataset:
            imdb_pairs = gen_MP_Edits_IMDB(imdb, num_lines, fill_models=['BART'], n_grams=[1, 2, 3], nlp=nlp, bart=bart, scorer=scorer)
        if 'MNLI' in dataset:
            mnli_hypos, mnli_prems = gen_MP_Edits_NLI(mnli_train, num_lines, fill_models=['BART'], n_grams=[1, 2, 3], nlp=nlp, bart=bart, scorer=scorer, filename = 'output/' + str(num_lines) + '_mnli')
        if 'SNLI' in dataset:
            snli_hypos, snli_prems = gen_MP_Edits_NLI(snli_train, num_lines, fill_models=['BART'], n_grams=[1, 2, 3], nlp=nlp, bart=bart, scorer=scorer, filename = 'output/' + str(num_lines) + '_snli')
    
    ## TASK: PREDICTING TAGS ON MASK-FILLED EXAMPLES.
    if data_action == 'Predict':
        if 'IMDB' in dataset:
            imdb_model = ClassificationModel("bert", "bert-imdb/checkpoint-15000-epoch-3", use_cuda = use_cuda)
            imdb_pairs = pd.read_csv('output/' + str(num_lines) + '_imdb.csv')
            tagged_imdb = predict_on_MP_imdb(imdb_pairs, imdb_model, len(imdb_pairs))
            imdb_contrast = tagged_imdb[tagged_imdb['label-changed'] == True]

            imdb_contrast.to_csv('output/' + str(num_lines) + '_imdb_contrast.csv')
            tagged_imdb.to_csv('output/' + str(num_lines) + '_imdb_tagged.csv')

        if 'MNLI' in dataset:
            # reading
            MNLI_Roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
            if torch.cuda.is_available():
                MNLI_Roberta.cuda()

            #MNLI_model = ClassificationModel("roberta", "roberta-mnli/checkpoint-146688-epoch-3", use_cuda = use_cuda)
            mnli_hypos = pd.read_csv('output/' + str(num_lines) + '_mnli_hypos.csv')
            mnli_prems = pd.read_csv('output/' + str(num_lines) + '_mnli_prems.csv')

            # tagging
            mnli_tagged_hypos, mnli_contrast_hypos = predict_on_MP_NLI(mnli_hypos, MNLI_Roberta, len(mnli_hypos), 'hypothesis')
            mnli_tagged_prems, mnli_contrast_prems = predict_on_MP_NLI(mnli_prems, MNLI_Roberta, len(mnli_prems), 'premise')
            mnli_tagged_hypos.to_csv('output/' + str(num_lines) + '_mnli_tagged_hypos.csv')
            mnli_tagged_prems.to_csv('output/' + str(num_lines) + '_mnli_tagged_prems.csv')

            mnli_contrast_hypos.to_csv('output/' + str(num_lines) + '_mnli_contrast_hypos.csv')
            mnli_contrast_prems.to_csv('output/' + str(num_lines) + '_mnli_contrast_prems.csv')

            compute_metrics(mnli_tagged_hypos)
            compute_metrics(mnli_tagged_prems)

        if 'SNLI' in dataset:
            # reading 
            MNLI_Roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
            MNLI_Roberta.eval()
            if torch.cuda.is_available():
                MNLI_Roberta.cuda()

            #SNLI_model = ClassificationModel("roberta", "roberta-snli/checkpoint-114291-epoch-3", use_cuda = use_cuda)
            snli_hypos = pd.read_csv('output/' + str(num_lines) + '_snli_hypos.csv')
            snli_prems = pd.read_csv('output/' + str(num_lines) + '_snli_prems.csv')

            # tagging
            snli_tagged_hypos, snli_contrast_hypos = predict_on_MP_NLI(snli_hypos, MNLI_Roberta, len(snli_hypos), 'hypothesis')
            snli_tagged_prems, snli_contrast_prems = predict_on_MP_NLI(snli_prems, MNLI_Roberta, len(snli_prems), 'premise')
            snli_tagged_hypos.to_csv('output/' + str(num_lines) + '_snli_tagged_hypos.csv')
            snli_tagged_prems.to_csv('output/' + str(num_lines) + '_snli_tagged_prems.csv')

    # compute metrics and make figures
    # compute_metrics(tagged_df)
    # similarity_figures(tagged_df)

# Usage: python3.8 MP-Edit.py <dataset> <number of lines> <which action to take>

if __name__ == '__main__':
    if parse_inputs():
        dataset, num_lines, data_action = parse_inputs()
        main(dataset, num_lines, data_action)


# # generic fill in masking pipeline.
# def fill_in_mask_BERT(input_sentence, model, tokenizer):
#     input = tokenizer.encode(input_sentence, return_tensors="pt")
#     mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]
#     token_logits = model(input).logits
#     mask_token_logits = token_logits[0, mask_token_index, :]
#     top_token = torch.topk(mask_token_logits, 1, dim=1).indices[0].tolist()
#     new_token = tokenizer.decode([top_token[0]])
#     output_sentence = input_sentence.replace(tokenizer.mask_token, new_token)

#     return output_sentence, new_token


# # BART specific fill in masking pipeline
# def fill_in_mask_BART(input_sentence, model, tokenizer):
#     input = tokenizer([input_sentence], return_tensors='pt')['input_ids']
#     mask_token_index = (input[0] == tokenizer.mask_token_id).nonzero()[0].item()  # pick first <mask>
#     token_logits = model(input, return_dict=True).logits
#     probs = token_logits[0, mask_token_index].softmax(dim=0)
#     values, predictions = probs.topk(5)
#     for token in tokenizer.decode(predictions).split():
#         new_token = token
#         if token not in string.punctuation:  # prefer not punctuation
#             break
#     output_sentence = input_sentence.replace(tokenizer.mask_token, new_token)

#     return output_sentence, new_token

# # fine tune pretrained bert model
# def train_IMDB(df, n_epoch):
#     ### Fine Tuning Classification Model on IMDB Data
#     df_train, df_valid = train_test_split(df, test_size=0.2)

#     args = {
#         'fp16': False,
#         'num_train_epochs': n_epoch,
#         'overwrite_output_dir': True,
#         'learning_rate': 1e-5,
#     }

#     if torch.cuda.is_available():
#         # bert base pretrained classification
#         imdb_model = ClassificationModel('bert', 'bert-base-cased', use_cuda=True, args=args)
#     else:
#         imdb_model = ClassificationModel('bert', 'bert-base-cased', use_cuda=False, args=args)

#     imdb_model.train_model(df_train, output_dir='bert-imdb')
#     result, model_outputs, wrong_predictions = imdb_model.eval_model(df_valid)

#     print((result['tp'] + result['tn']) / (result['tp'] + result['tn'] + result['fp'] + result['fn']))

#     return imdb_model


test = pd.read_csv('MP-Edit/output/MNLI/840_mnli_tagged_hypos.csv')

switches = zip(test['orig-label'], test['new-label'])
switch_changes = Counter(switches)
print(switch_changes)
