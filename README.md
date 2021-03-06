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

## Results

For num_lines = 840, MNLI label flips on hypothesis mask-infilling.

|  Original ->  | Entailment | Neutral | Contradiction |
|:-------------:|:----------:|:-------:|:-------------:|
|   Entailment  |    5181    |   315   |      387      |
|    Neutral    |    1472    |   5734  |      763      |
| Contradiction |     850    |   432   |      4828     |

## TO DO:

12-16-20
- [ ] Setup project on Greene cluster, gain more GPU memory.
- [ ] Fine-tune for MNLI prem+hypo hypo mask infilling on BART Large
	- [ ] Test various mask infill schemes, make sure compatible with fill_mask function.
	- [ ] Compare Losses, PPLs, training times. 
- [ ] Produce word embeddings/distance related visualization
- [ ] Decide between BERT Score and W2V embeddings, implement prem+hypo distance. 

11-25-20
- [x] Fine-tune prem+hypo mask infilling using BART-Base and Vishakh's sample code
	- [ ] Reimplement using task == Denoising, other masking strategies
- [ ] Generate new examples using fine-tuned BART
- [x] Reorganize Github and File Directory
- [x] Fix various bugs (similarity measures, string replacement by index, etc.)
- [x] Create Utility Functions, separate from main script

11-11-20
- [x] Conditional Generation, Premise + Masked Hypothesis
- [x] Implement Top K Sampling (10)
- [x] Use LargeBART for in-fill, LargeRoBERTA (fine-tuned on MNLI) for Classification
- [x] Fixed mask-replacement, from string match to index based removal.
- [ ] Finetune BART on MNLI for mask-infill (PROBLEM WITH FAIRSEQ IMPLEMENTATION)
- [ ] Top-p nucleus sampling (NOT IN FAIRSEQ IMPLEMENTATION)
- [ ] Integrate SSG2 Java for IMDB Sentiment Analysis (LIT tool)

11-4-20
- [x] Create flexible command line Train-Generate-Predict data pipeline
- [x] Setup Github and data folders
- [x] Reconfigure Transformers architecture for SNLI
- [x] Find clusters of mask-change and sentence-change occurences
- [ ] Create gradient policy for mask-infilling (positional) using label flip as reward

10-29-20
- [x] Develop sentence quality filter
- [x] Create ngram masking function
- [x] Use BART for ngram infilling
- [x] Add SNLI data set
- [x] Figure out CILVR cluster Python+CUDA Setup
- [ ] Fine-tune BART infill policy with classification data (not needed, hard enough to generate label flip)
