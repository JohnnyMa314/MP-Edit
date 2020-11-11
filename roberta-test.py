import torch
from fairseq.data.data_utils import collate_tokens

roberta = torch.hub.load('', 'roberta.large.mnli')
roberta.eval()
if torch.cuda.is_available():
	roberta.cuda()

batch_of_pairs = [
    ['Roberta is a heavily optimized version of BERT.', 'Roberta is not very optimized.'],
    ['Roberta is a heavily optimized version of BERT.', 'Roberta is based on BERT.'],
    ['potatoes are awesome.', 'I like to run.'],
    ['Mars is very far from earth.', 'Mars is very close.'],
]

batch = collate_tokens(
    [roberta.encode(pair[0], pair[1]) for pair in batch_of_pairs], pad_idx=1
)

logprobs = roberta.predict('mnli', batch)
print(logprobs)
print(logprobs.argmax(dim=1))
print(logprobs[logprobs.argmax(dim=1)]