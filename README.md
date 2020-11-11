## Counterfactual explanation for NLU models

Team:
- Johnny Ma 
- Nitish Joshi
- He He
- Sam Bowman (as of 11-4-20)

Minimal pairs is a nice way to evaluate models and expose the spurious correlations learned by the model.

Can we use counterfactual examples as a lense to inspect/debug models?
This requires a generator that can automatically generate “counterfactual” examples (“factual” w.r.t. the model. We don’t know if the gold label would change.).

We can learn such a generator (or rather an editor) by masking and infilling.
Specifically, we have a masking policy that chooses which words to mask in a sentence and a language model to fill in the masks.

The language model can be pre-trained so we just need to learn the masking policy, which can be done by policy gradient with loss function.

Once we have the generator, we can cluster patterns that flip the model prediction to understand the model behavior. 

Finally, we will crowdsource labels for correct and incorrect label flips, which allows us to augment our training data with a model-in-the-loop generated counterfactuals. 

## Running MP-Edit.py

MP-Edit.py takes three arguments: `dataset`, `num_lines`, and `data_action`. 

`Datasets` available are: 
- `IMDB`: IMDB Sentiment Classification
- `SNLI`: SNLI Entailment Task
- `MNLI`: MNLI Entailment Task

`Number of lines` determines the number of data instances to generate minimal pairs for, iterating over content words (NOUN, VERB, ADJ) and their surrounding ngrams (1,2,3). 

`Data actions` available are:
- `Train`: Fine-tune transformer model on dataset
- `Generate`: Generate minimal pair edits using BART mask-infilling
- `Predict`: Run prediction model over generated sentence edits.

e.g. `python3.8 MP-Edit.py MNLI 1000 Generate`

The outputted datasets or models will be found in the working directory under 'output' and 'models' respectively. 

## TO DO:

11-4-20
- [x] Create flexible command line Train-Generate-Predict data pipeline
- [x] Setup Github and data folders
- [x] Reconfigure Transformers architecture for SNLI
- [ ] Find clusters of mask-change and sentence-change occurences
- [ ] Create gradient policy for mask-infilling (positional) using label flip as reward

10-29-20
- [x] Develop sentence quality filter
- [x] Create ngram masking function
- [x] Use BART for ngram infilling
- [x] Add SNLI data set
- [x] Figure out CILVR cluster Python+CUDA Setup
- [ ] Fine-tune BART infill policy with classification data (not needed, hard enough to generate label flip)
