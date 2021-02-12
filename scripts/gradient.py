# import packages
import torch
import numpy as np
import json
import pandas as pd
from fairseq.models.bart import BARTModel
import random
from fairseq.data.data_utils import collate_tokens
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
            if loaded_example["label"] not in LABEL_MAP:
                continue
            loaded_example["label"] = LABEL_MAP[loaded_example["label"]]
            data.append(loaded_example)
            
        random.seed(1)
        random.shuffle(data)

    f.close()
    return data

# import test data

mnli = pd.DataFrame(load_nli_data('data/multinli_1/mnli_bart_test_gold_labeled.jsonl'))
print(mnli.columns)

# open test model
model = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
model.eval()
if torch.cuda.is_available():
	model.cuda()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print('cuda:0')

torch.manual_seed(123)
np.random.seed(123)

for p in model.parameters():
    param_norm = p.grad.data.norm(2)
    total_norm += param_norm.item() ** 2
total_norm = total_norm ** (1. / 2)

# # convert data to tensors
# masked_pairs = []
# premise = mnli['sentence1']
# hypothesis = mnli['sentence2']
# label = mnli['label']

# tokens = model.encode(premise[0], hypothesis[0])
# print(tokens)

# print(model.predict('mnli', tokens))


# # # run tensors through cap
# # last_layer_features = model.extract_features(tokens)
# # print(last_layer_features)
# # all_layers = model.extract_features(tokens, return_all_hiddens=True)
# # print(all_layers[0])


# # decode tensors to strings

# # sort by gradient

# go down list constructing spans until reach 



# label = mnli['label']
# for i in range(len(mnli)):
# 	masked_pairs.append([premise[i], hypothesis[i]])

# first = model.encode(premise[0])

# encodings = [model.encode(str(pair[0]), str(pair[1])) for pair in masked_pairs]
# mask_batch = collate_tokens(encodings, pad_idx=1)
# print(mask_batch)
