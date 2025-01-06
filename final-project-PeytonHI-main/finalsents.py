import json
import ollama
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import requests
from bs4 import BeautifulSoup
import requests
import spacy
from sklearn.metrics import classification_report
import regex as re


# query and return titles associated with query
def search_wikipedia(query):
    # Search Wikipedia
    search_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "utf8": 1,
        "srlimit": 5

    }

    response = requests.get(search_url, params=params)
    assert response.status_code == 200, f'Error querying wikipedia {response.status_code}'
    print("Successful wiki query")
    search_results = response.json()
    
    titles = []
    for result in search_results.get("query", {}).get("search", []):
        titles.append(result)
    
    return titles

# get page data by passing in each title and return html data for page
def get_wikipedia_page(title):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "parse",
        "page": title,
        "prop": "text",
        "format": "json",
        "formatversion": 2,
        "utf8": 1
    }

    response = requests.get(url, params=params)
    page_data = response.json()

    # Extract the page summary
    if 'parse' in page_data:
        return page_data['parse']['text']
    else:
        return None

# parse html with BeautifulSoup
def extract_html_info(html_content):
    if html_content is None:
      return None
    soup = BeautifulSoup(html_content, "html.parser")
    
    # get paragraphs
    paragraphs = [p.get_text().strip() for p in soup.find_all("p") if p.get_text().strip()]
    
    return paragraphs
            
# create labels based on given claims and documents using ollama
def generate_label(claim, documents):
    response = ollama.chat(
        model="llama3.2:1b",
        messages=[
            {
                "role": "user",
                "content": f"Given the following documents determine if the following claim is true. \nDocuments: {documents}\nClaim: {claim}\nOnly reply with SUPPORTS, REFUTES, or NOT ENOUGH INFO and nothing else. Choose the label with the best evidence."
            }
        ]
    )
    return response

# sim index between claim and all content (paragraphs), 
def is_similar_query(model, claim, content):
    device = torch.device("cpu")
    similarity_index = {}
    claim_tensor = model.encode(claim, convert_to_tensor=True)
    # claim_tensor = claim_tensor.unsqueeze(0)
    claim_tensor.to(device)
    # print('claim_tensor shape :', claim_tensor.shape)

    text = ' '.join(content)
    sentences = re.split(r'(?<=[.!?]) +', text)
    for sent in sentences:
        sent_tensor = model.encode(sent, convert_to_tensor=True)
        sent_tensor.to(device)    
        # print('content shape :', sent_tensor.shape)
        similarity = util.pytorch_cos_sim(claim_tensor, sent_tensor)
        # print('sim shape :', similarity.shape)
        similarity = similarity.item()

        similarity_index[sent] = similarity

    return similarity_index


def getTopSentences(sentences, k=3):
    sorted_dict_list = []
    for sent_score_dict in sentences:
        sorted_dict = sorted(sent_score_dict, key=sent_score_dict.get, reverse=True)[:k]
        sorted_dict_list.append(sorted_dict)

    return sorted_dict_list

def main():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model = model.to('cpu')

    # load spacy model
    nlp = spacy.load("en_core_web_sm")

    with (open(r"C:\Users\peyto\Desktop\school24\497\final\data\fever_train.jsonl", encoding='UTF-8')) as f:
        lines = f.readlines()
    dataList = [json.loads(line) for line in lines]
    datamini = dataList[:1000]
    
    refs_binary = []
    preds_binary = []
    count = 0
    cv_pair = []
    for data in tqdm(datamini):
        claim = data['claim']
        true_label = data['label']
        label_map = {'NOT ENOUGH INFO': 0, 'SUPPORTS': 1, 'REFUTES': 2}
        refs_binary.append(label_map[true_label])

        # pass claim to spacy
        doc = nlp(claim)

        # extract keywords (nouns, proper nouns, etc.)
        keywords = [token.text for token in doc if token.pos_ in ("NOUN", "PROPN")]
        query = " ".join(keywords)  

        # grab first 6 wiki results and check their similarity with claim then add to dict
        wiki_search_titles = search_wikipedia(query)
        similar_list = []
        for data_point in wiki_search_titles[:5]:
            title = data_point.get('title')
            if title is None:
                break
            print("Wiki title: ", title)

            html_evidence = get_wikipedia_page(title) # retrieve html page
            evidence_doc = extract_html_info(html_evidence) # extract paragraphs (docs)
            if evidence_doc is None:
                break
            # sentence:score
            sentence_scores = is_similar_query(model, claim, evidence_doc) # check similarity between claim and docs
            if sentence_scores is None:
                break
            
            similar_list.append(sentence_scores)

        top_k_sents = getTopSentences(similar_list) # select top 3 articles with highest cosine sim scores

        response = generate_label(claim, top_k_sents) 
        pred_label = response["message"]["content"]
        pred_label = pred_label.strip()

        if re.match(r'^\s*[Nn]', pred_label):
            pred_label = 'NOT ENOUGH INFO'
        elif re.match(r'^\s*[Ss]', pred_label):
            pred_label = 'SUPPORTS'
        elif re.match(r'^\s*[Rr]', pred_label):
            pred_label = 'REFUTES'
    
        preds_binary.append(label_map[pred_label])
        cv_pair.append((claim, pred_label))
        data_length = len(datamini)
        count += 1
        print(f"Claim added: {count}/{data_length}")

    with open("custom-multiQSents.jsonl", "w") as fout:
      for id, (claim, pred_label) in enumerate(cv_pair):
          j = {'id': id,'label': pred_label, 'claim': claim}
          print(json.dumps(j), file=fout)

    print("Document created")

    print(classification_report(refs_binary, preds_binary, target_names=label_map.keys()))

if __name__ == '__main__':
    main()