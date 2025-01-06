def tokenize_whitespace(text: str) -> list[str]:
    """
    Tokenize by splitting on whitespace.
    """
    tokens = text.split(" ")
    return tokens

def featurize_bigram(tokens):
    
    bigram_feats = {}
    for i in range(len(tokens) - 1):
        bigram = tuple(tokens[i:i+2])
        if bigram not in bigram_feats:
            bigram_feats[bigram] = 1
        else:
            bigram_feats[bigram] += 1

    return bigram_feats

def featurize_trigram(tokens):
    
    trigram_feats = {}
    for i in range(len(tokens) - 1):
        trigram = tuple(tokens[i:i+3])
        if trigram not in trigram_feats:
            trigram_feats[trigram] = 1
        else:
            trigram_feats[trigram] += 1

    return trigram_feats

def featurize_unigram(tokens):
    
    unigram_feats = {}
    for i in range(len(tokens) - 1):
        unigram = tuple(tokens[i: i + 1])
        if unigram not in unigram_feats:
            unigram_feats[unigram] = 1
        else:
            unigram_feats[unigram] += 1

    return unigram_feats

 # not working
def featurize_context_feats(tokens):
    context_feats = {}
    for context_feat in tokens:
        if "city" not in tokens:
                context_feats[context_feat] = 0
        else:
            if "city" in tokens:
                    context_feats[context_feat] = 0
                    context_feats[context_feat] += 1

    return context_feats

        
# def tokenize(text: str) -> list[str]:
#     raise NotImplementedError()


# def featurize(tokens: list[str]) -> dict:
#     raise NotImplementedError()
