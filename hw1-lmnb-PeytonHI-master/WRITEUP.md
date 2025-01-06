# Writeup

# PART 1:

Tokenizer used:
I'm simply splitting on whitespace and my generated token output seems to not make very much sense. Perhaps it is the way I am generating my greedy and top k tokens improperly and instead of looking at only the prior token, I need to make my ngram generating dynamic with the start context length given. Tbat way, I'll be generating next tokens based on every word prior, not just 1 word prior.

Perplexity data:
Train perplexity: 5679.426633308954
Dev perplexity: 933.8438641205556

This data is in line with the fact that my tokenization is not thorough. I am using simple tokens that split on whitespaces. If I were for instance to use BPE, my perplexities would be much lower.


Generation: top-k sampling with varying k values.
k=5: ['sherlock', 'holmes', 'took', 'it', 'up', 'and', 'examined', 'it.', 'one', 'of', 'the', 'provinces', 'of', 'my', 'kingdom']
k=10: ['sherlock', 'holmes', 'took', 'it', 'up', 'and', 'examined', 'it.', 'one', 'of', 'the', 'royal', 'brougham', 'rolled', 'down']
k=15: ['sherlock', 'holmes', 'took', 'it', 'up', 'and', 'opened', 'the', 'goose.', 'my', 'heart', 'turned', 'to', 'water,', 'for']

The lowest value of k at 5 seems to produce the most coherent sentence. Perhaps due to the limited pool of next most available options being the next most likely in a given sequence.

# Part 2:


MODEL     PRECISION             RECALL                 F1
bigram    0.2222222222222222    0.00966183574879227    0.01851851851851852

trigram   0.07692307692307693   0.001610305958132045   0.003154574132492114

unigram   0.3181818181818182    0.18035426731078905    0.2302158273381295

Features tried include unigrams, bigrams, and trigrams. Unigram produced the highest f1 score albeit still closer to 0 then 1, thus indicating my model isn't prediciting accurately.




