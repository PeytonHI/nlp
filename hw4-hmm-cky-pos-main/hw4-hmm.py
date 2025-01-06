from collections import Counter
import math

# Author: Jaden, Peyton

def load_conllu(file):
    data = []
    with (open(file) as fin):
        current = []
        for line in fin:
            arr = line.strip('\n').split('\t')

            if arr[0] == "" or arr[0].startswith("#") or arr[0].__contains__('-') or arr[0].__contains__('.'):
                if current != []:
                    data.append(current)
                    current = []
                continue

            d = {
                "id": int(arr[0]),
                "form": arr[1],
                "lemma": arr[2],
                "upos": arr[3],
                "xpos": arr[4],
                "feats": arr[5],
                "head": arr[6],
                "deprel": arr[7],
                "deps": arr[8],
                "misc": arr[9]
            }
            current.append(d)

        if current != []:
            data.append(current)
    return data


class HMMClassifier:
    def __init__(self):
        self.trained = False
        self.tags = set()
        self.words = set()
        self.tag_counts = Counter()
        self.word_tag_counts = Counter()
        self.tag_transition_counts = Counter()
        self.initial_tag_counts = Counter()
        self.total_sentences = 0

    def train(self, data):
        for sentence in data:
            self.total_sentences += 1

            # count initial tags
            first_tag = sentence[0]['upos']
            self.initial_tag_counts[first_tag] += 1

            for i, token in enumerate(sentence):
                current_tag = token['upos']
                current_word = token['form'].lower()
            
                self.tags.add(current_tag)
                self.words.add(current_word)

                # count tag occurrences
                self.tag_counts[current_tag] += 1

                # count word-tag co-occurrences
                self.word_tag_counts[(current_word, current_tag)] += 1

                # count tag transitions 
                if i < len(sentence) - 1:
                    next_tag = sentence[i + 1]['upos']
                    self.tag_transition_counts[(current_tag, next_tag)] += 1

        self.trained = True

    def transition(self, t_i, t_iminus1):
        # p(t_i | t_iminus1) 
        numerator = self.tag_transition_counts[(t_iminus1, t_i)]
        denominator = self.tag_counts[t_iminus1] + len(self.tags)
        return numerator / denominator if denominator > 0 else 1e-10

    def emission(self, w_i, t_i):
        # p(w_i | t_i)
        numerator = self.word_tag_counts[(w_i, t_i)]
        denominator = self.tag_counts[t_i] + len(self.words)
        return numerator / denominator if denominator > 0 else 1e-10

    def initial(self, t):
        # p(t) 
        numerator = self.initial_tag_counts[t]
        denominator = self.total_sentences + len(self.tags)
        return numerator / denominator if denominator > 0 else 1e-10

    def predict(self, sentence: list[str]):
        # viterbi algorithm
        if not self.trained:
            raise Exception("must train first!")

        # convert words to lowercase for even basis
        # sentence = [w.lower() for w in sentence]
        
        viterbi = [[0.0 for _ in range(len(sentence))] for _ in range(len(self.tags))]
        backpointers = [['' for _ in range(len(sentence))] for _ in range(len(self.tags))]

        # convert tags to list for consistent indexing
        tags_list = sorted(list(self.tags))
    
        for i, tag in enumerate(tags_list):
            initial_prob = self.initial(tag)
            emission_prob = self.emission(sentence[0], tag)

            # initial prob or emission is 0
            if initial_prob <= 0 or emission_prob <= 0:
                viterbi[i][0] = float('-inf')  # neg infinity if prob is zero
            else:
                viterbi[i][0] = math.log(initial_prob) + math.log(emission_prob)
            backpointers[i][0] = None
        
        for t in range(1, len(sentence)):
            for i, current_tag in enumerate(tags_list):
                max_prob = float('-inf')
                best_previous_tag = None

                # for each possible previous tag
                for j, previous_tag in enumerate(tags_list):
                    transition_prob = self.transition(current_tag, previous_tag)
                    emission_prob = self.emission(sentence[t], current_tag)
                    if transition_prob > 0 and emission_prob > 0:
                        prob = (
                                viterbi[j][t - 1] +
                                math.log(transition_prob) +
                                math.log(emission_prob)
                        )
                    else:
                        prob = float('-inf') 

                    if prob > max_prob:
                        max_prob = prob
                        best_previous_tag = previous_tag

                viterbi[i][t] = max_prob
                backpointers[i][t] = best_previous_tag

        # get best final tag
        final_max_prob = float('-inf')
        final_best_tag = None
        for i, tag in enumerate(tags_list):
            if viterbi[i][-1] > final_max_prob:
                final_max_prob = viterbi[i][-1]
                final_best_tag = tag

        # follow backpointers to get best parse
        best_parse = [final_best_tag]
        current_tag = final_best_tag
        current_tag_idx = tags_list.index(current_tag)

        for t in range(len(sentence) - 1, 0, -1):
            current_tag = backpointers[current_tag_idx][t]
            current_tag_idx = tags_list.index(current_tag)
            best_parse.insert(0, current_tag)

        return viterbi, backpointers, best_parse


def main():
    data = load_conllu("\data\example.conllu")
    # print(data)

    model = HMMClassifier()
    model.train(data)

    print("Transition Probabilities")
    for t_iminus1 in sorted(model.tags):
        for t_i in sorted(model.tags):
            print(f"p({t_i} | {t_iminus1}) = {model.transition(t_i, t_iminus1)}")

    print("Emission Probabilities")
    for t in sorted(model.tags):
        for w in sorted(model.words):
            print(f"p({w} | {t}) = {model.emission(w, t)}")

    viterbi, backpointers, best_parse = model.predict("the old man".split())
    print(best_parse)


if __name__ == "__main__":
    main()
