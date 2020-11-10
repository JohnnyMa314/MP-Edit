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

## TO DO:

11-4-20
- [x] Create flexible command line Train-Generate-Predict data pipeline
- [x] Setup Github and data folders
- [ ] Reconfigure Transformers architecture for SNLI
- [ ] Find clusters of mask-change and sentence-change occurences
- [ ] Create gradient policy for mask-infilling (positional) using label flip as reward

10-29-20
- [x] Develop sentence quality filter
- [x] Create ngram masking function
- [x] Use BART for ngram infilling
- [x] Add SNLI data set
- [x] Figure out CILVR cluster Python+CUDA Setup
- [ ] Fine-tune BART infill policy with classification data (not needed, hard enough to generate label flip)
