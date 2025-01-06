from lm import tokenize, NGramLanguageModel

def main():
    with open("data/sherlock.txt", encoding ="utf-8") as fin:

        sherlock = fin.read()
    with open("data/ttc.txt",  encoding ="utf-8") as fin:
        ttc = fin.read()
    
    sherlock_tokens = tokenize(sherlock)
    ttc_tokens = tokenize(ttc)

    lm = NGramLanguageModel(4)
    lm.train(sherlock_tokens)
    lm.prob(sherlock_tokens)

    greedy_tokens = lm.generate_greedy(start_context=["sherlock", "holmes", "took"], length=8)
    print(greedy_tokens)
    top_k_tokens = lm.generate_topk(start_context=["sherlock", "holmes", "took"], length=15, k=15)
    print(top_k_tokens)
    print("Train perplexity:", lm.perplexity(sherlock_tokens))
    print("Dev perplexity:", lm.perplexity(ttc_tokens))


if __name__ == "__main__":
    main()
