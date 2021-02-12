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
import resource
from fairseq.models.bart import BARTModel
from fairseq.data.data_utils import collate_tokens
import datetime
import re
import gc
import json


### Usage: python3.8 MP-Edit.py <dataset> <number of lines> <which action to take> <which generation model to use> ###

def load_nli_data(path):
    """
    Load MultiNLI or SNLI data.
    """
    LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2}
    
    data = []
    with open(path) as f:
        for line in f:
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue
            loaded_example["label"] = LABEL_MAP[loaded_example["gold_label"]]
            data.append(loaded_example)
            
        random.seed(1)
        random.shuffle(data)

    f.close()
    return data

def mask_sentence(sentence, masking, n_grams):
	
	mask_indices = []
	masked_hypos = []

	for token in sentence:
		if masking == 'data-slices':
			## predefine interesting data points
			# find "interesting" RTE data slices, add as filter
			# use slices defined in paper. simple text matching.

			temporal_prepositions = ["after", "before", "past"]
			comparative_words = ["more", "less", "better", "worse", "bigger", "smaller"]
			quantifiers = ["all", "some", "none"]
			negation_words = ["no", "not", "none", "no one", "nobody", "nothing", "neither", "nowhere", "never", "hardly", "scarcely", "barely", "doesnt", "isnt", "wasnt", "shouldnt", "wouldnt", "couldnt", "wont", "cant", "dont"]

			# if any slice words are in premise+hypothesis
			total_SF = temporal_prepositions + comparative_words + quantifiers + negation_words

			if token.text in total_SF:
				mask_indices.append(token.i)

		if masking == 'content-words':
			# simple content word matching by Part of Speech (POS)
			if token.pos_ == 'NOUN' or token.pos_ == 'ADJ' or token.pos_ == 'VERB' and token.text == '':  # spacy POS for content
				mask_indices.append(token.i)

		if masking == 'gradient':
			print('yes')

	# for the marked tokens' indices, generate ngrams around the token. 
	for ind in mask_indices:			
		for N in n_grams:
			ngram_range = range(ind - N + 1, ind+1)
			for i in ngram_range:
				ngram = sentence[i:i + N] # to make sure we are getting N-Grams at ends of sentences
				if len(ngram) == N:
					masked_hypo = re.sub(r'\s+([?.!"])', r'\1', (sentence[:i].text + ' <mask> ' + sentence[i+N:].text).strip()) # remove spaces before punctuation
					masked_hypos.append(masked_hypo)

	# removes duplicates
	return list(set(masked_hypos)) 


def create_masked_pair(masked_hypo, row, prompting):
	# return concatenated prompt + premise + </s> + masked_hypo
	classes = ['entailment', 'contradiction', 'neutral']

	if prompting == 'gold-label':
		prepend_label = row['label']

	if prompting == 'model-label':
		prepend_label = row['model_label']
		
	classes.remove(prepend_label) 
	new_label = random.sample(classes, 1)[0] # flip to one of the other two classes.

	masked_pair = 'label: ' + new_label + '. input: ' + row['premise'] + ' </s> ' + masked_hypo

	return masked_pair, new_label


def get_mask_infill_candidates(masked_pair, sampling, bart):
	# get sentences and probabilities for 20 candidate fills.
	bart.cuda()
	if sampling == 'beam':
		candidates = [(sent, prob.item()) for sent, prob in 
			bart.fill_mask(masked_pair, topk = 10, beam=10, match_source_len=False)]

	if sampling == 'topp':
		candidates = [(sent, prob.item()) for sent, prob in 
			bart.fill_mask(masked_pair, sampling = True, sampling_topp = 0.90, sampling_topk = 10, match_source_len=False)]

	if sampling == 'diverse-beam':
		candidates = [(sent, prob.item()) for sent, prob in 
			bart.fill_mask(masked_pair, topk = 10, beam = 10, diversity_rate = 0.3, match_source_len = False)]
									
	return candidates


def gen_MPs_NLI(df, num_lines, n_grams, bart, fill_model, masking, prompting, sampling):
	# load relevant nlp packages
	nlp = spacy.load('en_core_web_lg')
	scorer = BERTScorer(lang="en")

	## pre-define dataframe
	cond_gens = pd.DataFrame(columns=['line-num', 'pred-model', 'fill-model', 'tokens-masked', 'prepend-model', 'sampling-strategy',
								 'premise', 'hypothesis', 'mask-filled', 'token_changes', 'fill_prob', 'depth', 'Word2Vec-Score',
								 'Bert-Score', 'gold-label', 'prepend-label', 'targeted-label'])  # setup

	# for each data instance, get sentences, add prompts, mask words, sample from mask-infills.
	# counter is based on data instances, many of which will be immediately skipped over. 

	for index, row in tqdm(df[0:int(num_lines)].iterrows(), total=df[0:int(num_lines)].shape[0]):
		# change hypothesis
		text = row['hypothesis']  
		sentence = nlp(text)
		gen_hypos = [] # keep a track of filled sentences to not duplicate.

		if 10 < len(sentence) < 30 and 10 < len(nlp(row['premise'])) < 30:  # if either sentence has more than 10 and less than 30 tokens.
			masked_hypos = mask_sentence(sentence, masking, n_grams)

			for masked_hypo in masked_hypos:

				masked_pair, new_label = create_masked_pair(masked_hypo, row, prompting)

				try:
					candidates = get_mask_infill_candidates(masked_pair, sampling, bart)
				except:
					print("ERROR with invalid literal for int() with base 10:")
					continue
				
				print(row)

				# remove premise conditional, other artifacts (may be unnecessary)
				tmp_sents = []
				sents = [sent.replace('label: ' + new_label + '. input: ', '').replace('label: ' + new_label + '. input:', '').replace('』', '').replace('├', '').replace(row['premise'] + ' </s> ', '').replace('</s>', '').replace('<s>', '').replace(row['premise'], '').replace('<br>', '').replace('<s>', '').replace('<', '').replace('>', '').replace('(', '').replace('[', '').replace('│','').replace(' , ', ', ').strip() for sent, prob in candidates]
				
				# remove symbols and numbers that appear at beginning of phrases (may be unnncessary). 
				for sent in sents:
					while sent and not sent[0].isalpha():
						sent = sent[1:]
					tmp_sents.append(sent)
				sents = tmp_sents

				# turn into readable probabilities
				probs = [prob for sent, prob in candidates]
				probs = softmax(probs)

				# if nothing is generated (somehow)
				if len(sents) < 1: 
					continue
				
				# go down the list of top candidates until we find an acceptable generation
				depth = 0 # track how deep the sampling goes
				print(f'Original Sentence: {sentence.text} \n')
				for sent in sents:
					depth = depth + 1
					print('Fillled Sentence: ' + sent + '\n')
					new_token = ' '.join(
						list((Counter(sentence.text.split(' ')) - Counter(
							sent.split(' '))).keys()))  # get new words (NEEDS TO BE PROPERLY IMPLEMENTED)


					# removes duplicates, sentence infills, etc.
					if sentence.text.lower() != sent.lower() and len(sent) < len(sentence.text)*1.5 and sent not in gen_hypos:
						filled_sent = sent
						token_prob = str(round(probs[sents.index(sent)]*100, 2)) + '%'
						gen_hypos.append(filled_sent)
						break

				if sentence.text.lower() == sent.lower() or depth == 10: # final duplicate check
					continue
	
				# NEEDS TO BE PROPERLY IMPLEMENTED
				new_token = ' '.join(list((Counter(filled_sent.split(' ')) - Counter(
						masked_hypo.split(' '))).keys()))  # get new tokens

				cond_gens = cond_gens.append({'line-num': index,
										'pred-model': 'RoBERTa-MNLI',
										'fill-model': fill_model,
										'tokens-masked': masking,
										'prepend-model': prompting,
										'sampling-strategy': sampling,
										'premise': row.premise,
										'hypothesis': sentence.text,
										'mask-filled': filled_sent,
										'token_changes': new_token,
										'fill_prob': token_prob,
										'depth': depth,
										#'n_gram': N,
										'Word2Vec-Score': nlp(sentence.text).similarity(nlp(filled_sent)),
										#'token-similarity': nlp(ngram.text).similarity(nlp(new_token)),
										'Bert-Score': scorer.score([sentence.text], [filled_sent])[2].item(),
										'gold-label': row.label,
										'prepend-label': row.label,
										'targeted-label': new_label}, ignore_index=True)

		# save after each example just in case. 
		cond_gens.to_csv('output/' + str(num_lines) + '_' + fill_model + '_' + masking + '_' + prompting + '_' + sampling + '_MNLI.csv')

	return cond_gens

# predicting 
def predict_on_MP_NLI(df, NLI_model, num_lines):
	# filling in model predictions and probabilities for each sentence
	masked_pairs = []
	orig_pairs = []

	# fill in lists of premise-hypo pairs for original and masked
	# must be list of pairs for fairseq to do Entailment task
	masked_pairs = [list(x) for x in zip(df['premise'], df['mask-filled'])]
	orig_pairs = [list(x) for x in zip(df['premise'], df['hypothesis'])]
   
	# generating batch predictions from fine tuned model
	mask_batch = collate_tokens([NLI_model.encode(str(pair[0]), str(pair[1])) for pair in masked_pairs], pad_idx=1)
	orig_batch = collate_tokens([NLI_model.encode(str(pair[0]), str(pair[1])) for pair in orig_pairs], pad_idx=1)

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

	# TEMPORARY SOLUTION TO BUGGY BART INFILLING, OR NOT REACHING DEEP INTO BEAM
	df = df[df['Word2Vec-Score'] > 0.8]
	df = df[df['Word2Vec-Score'] < 1]

	return df

# Parsing inputs from command line
def parse_inputs():
	process = True

	try:
		num_lines = sys.argv[1]
	except:
		print("Enter the number of lines to generate over as the second argument. Make sure data is in correct folder.")
		process = False

	try:
		model_name = sys.argv[2]
	except:
		print("Please enter which model to use to generate/classify examples.")
		process = False

	try:
		masking = sys.argv[3]
	except:
		print("Please enter a masking strategy (data slices, content words, gradient, token importance).")
		process = False

	try:
		prompting = sys.argv[4]
	except:
		print("Please enter which prompting is used to control generation (gold, predicted, control code, none)")
		process = False

	try:
		sampling = sys.argv[5]
	except:
		print("Please enter a sampling strategy (beam, topp, hamming, etc.")
		process = False

	if process:
		return (num_lines, model_name, masking, prompting, sampling)
	else:
		return False


def main(num_lines, model_name, masking, prompting, sampling):
	random.seed(1996)

	print(num_lines) # number of data insatnces over which to generate
	print(model_name) # which model (fine-tuned, none) to use as generator
	print(masking) # which masking strategy (content-words, data-slices)
	print(prompting) # which pre-pend strategy (gold-label, model-label)
	print(sampling) # which sampling strategy (beam, diverse-beam, topp (topp has some errors still))

	# if torch.cuda.is_available():
	# 	print("Cuda is on")

	# ## STEP 1: OPEN DATA 

	# # processing MNLI data
	# if prompting == 'gold-label':
	# 	mnli = pd.read_csv('data/multinli_1/mnli_bart_test_gold_labeled.csv')
	# 	mnli = mnli.sample(frac=1, random_state = 1996)

	# 	mnli_test = mnli[['sentence1', 'sentence2', 'label']] # specify that the predictions are on RoBERTA
	# 	mnli_test.columns = ["premise", "hypothesis", "label"]

	# if prompting == 'model-label':
	# 	mnli = pd.read_csv('data/multinli_1/mnli_bart_test_pred_labeled.csv')
	# 	mnli = mnli.sample(frac=1, random_state = 1996)

	# 	mnli_test = mnli[['sentence1', 'sentence2', 'label', 'RoBERTA_pred']] # specify that the predictions are on RoBERTA
	# 	mnli_test.columns = ["premise", "hypothesis", "label", 'model_label']

	# ## STEP 2: LOAD GENERATION MODEL
	# # loading NLP packages and mask-fill model
	# nlp = spacy.load('en_core_web_lg')
	# scorer = BERTScorer(lang="en")

	# if 'none' in model_name:
	# 	bart = torch.hub.load('pytorch/fairseq', 'bart.large')
	# if 'fine-tuned' in model_name:
	# 	if prompting == 'gold-label':
	# 		bart = BARTModel.from_pretrained('checkpoints/', checkpoint_file='checkpoint_best_gold.pt')
	# 	if prompting == 'model-label':
	# 		bart = BARTModel.from_pretrained('checkpoints/', checkpoint_file='checkpoint_best_model.pt')

	# bart.eval()
	# if torch.cuda.is_available():
	# 	bart.cuda()

	# ## STEP 3: GENERATE NEW HYPOTHESIZES BY MASK-INFILLING

	# mnli_pairs = gen_MPs_NLI(mnli_test, num_lines, n_grams=[1, 2, 3], bart=bart,
	# 	fill_model = model_name, prompting = prompting, masking = masking, sampling = sampling)
	
	## STEP 4: PREDICTING LABEL OF ORIGINAL AND MASK-INFILLED NLI PAIRS
	# loading model FTed on MNLI Entailment Task
	MNLI_Roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
	if torch.cuda.is_available():
		MNLI_Roberta.eval()
		MNLI_Roberta.cuda()

	mnli_pairs = pd.read_csv('./output/5000_fine-tuned_content-words_gold-label_diverse-beam_MNLI.csv')

	# tagging
	mnli_tagged_pairs = predict_on_MP_NLI(mnli_pairs, MNLI_Roberta, len(mnli_pairs))
	#mnli_tagged_pairs.to_csv('output/' + str(num_lines) + '_' + model_name + '_' + masking + '_' + prompting + '_' + sampling + '_mnli_cond_pairs_tagged.csv')
	mnli_tagged_pairs.to_csv('./output/5000_fine-tuned_content-words_gold-label_diverse-beam_mnli_cond_pairs_tagged.csv')




	mnli_pairs = pd.read_csv('./output/5000_fine-tuned_content-words_model-label_diverse-beam_MNLI.csv')
	mnli_tagged_pairs = predict_on_MP_NLI(mnli_pairs, MNLI_Roberta, len(mnli_pairs))
	mnli_tagged_pairs.to_csv('./output/5000_fine-tuned_content-words_model-label_diverse-beam_mnli_cond_pairs_tagged.csv')

if __name__ == '__main__':
	if parse_inputs():
		num_lines, model_name, masking, prompting, sampling = parse_inputs()
		main(num_lines, model_name, masking, prompting, sampling)
