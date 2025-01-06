import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertForSequenceClassification
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertForTokenClassification

# Author: Jaden, Peyton

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")


class POSDataset(Dataset):
    def __init__(self, tokenized_inputs, aligned_labels, tag2idx):
        self.tokenized_inputs = tokenized_inputs
        self.aligned_labels = aligned_labels
        self.tag2idx = tag2idx

    def __len__(self):
        return len(self.tokenized_inputs)

    def __getitem__(self, idx):
        input_ids = self.tokenized_inputs[idx]['input_ids'].squeeze(0)
        attention_mask = self.tokenized_inputs[idx]['attention_mask'].squeeze(0)
        labels = self.aligned_labels[idx]  # returns array of tags

        # convert tags to indices
        label_ids = [self.tag2idx[label] for label in labels]


        padding_length = len(input_ids) - len(label_ids)
        if padding_length > 0:
            label_ids += [-100] * padding_length  # padding with -100
        else:
            label_ids = label_ids[:len(input_ids)] 
            
        return {
        'input_ids': torch.tensor(input_ids),
        'attention_mask': torch.tensor(attention_mask),
        'labels': torch.tensor(label_ids)  
    }
    

def load_conllu(file):
    data = []
    with (open(file, encoding='UTF-8') as fin):
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


def build_data(data):
    tokenized_data = []
    aligned_labels = []

    for d in data:
        # for each sentence in the data
        sentence_arr = []
        pos_tags = []

        for w in d:
            # for each word in sentence, tokenize for pos tagging alignment
            pos_tags.append(w['upos'])
            sentence_arr.append(w['form'])
            # print("POS:", w['upos'])
            # print("WORD:", w['form'])

            # align the POS tag to the tokens (repeat POS tag for the same tokenized word)
            aligned_tags = []
            for word, pos in zip(sentence_arr, pos_tags):
                tokenized_word = tokenizer.tokenize(word)
                aligned_tags.extend([pos] * len(tokenized_word))

                # tokenize the sentence
        sentence = " ".join(sentence_arr)

        tokens = tokenizer(sentence, padding='max_length', truncation=True, return_tensors='pt')
        tokenized_data.append(tokens)

        aligned_labels.append(aligned_tags)
        # print(sentence_arr)
        # print(aligned_tags)

    return tokenized_data, aligned_labels


def generate_tag_ids(data):
    tags = set()
    for sentence in data:
        for word in sentence:
            tags.add(word['upos'])

    tag2idx = {}
    i = 0
    for tag in tags:
        tag2idx[tag] = i
        i += 1

    return tag2idx


def train(model, train_loader, val_loader):
    device = torch.device("cuda")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    for epoch in range(1):
        model.train()
        total_train_loss = 0
        total_batches = 0

        for batch in tqdm(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            # print(f'Input IDs Shape: {input_ids.shape}, Attention Mask Shape: {attention_mask.shape}, Labels Shape: {labels.shape}')
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            total_batches += 1

            print(f"Train Loss: {loss.item() / len(train_loader)}")

        print(f"Epoch {epoch + 1} Train Loss: {total_train_loss / len(train_loader)}")
        val_loss = evaluate_val(model, val_loader)
        print(f"Epoch {epoch + 1} Validation Loss: {val_loss}")


def evaluate_val(model, dataloader):
    device = torch.device("cuda")
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                # print(f'Input IDs Shape: {input_ids.shape}, Attention Mask Shape: {attention_mask.shape}, Labels Shape: {labels.shape}')

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    device = torch.device("cuda")
    
    train_data = load_conllu(r"C:\Users\peyto\Desktop\school24\497\hw4-jaden-main\data\en_ewt-ud-train.conllu")
    dev_data = load_conllu(r"C:\Users\peyto\Desktop\school24\497\hw4-jaden-main\data\en_ewt-ud-train.conllu")
    # print(train_data[0])

    t_tokenized_inputs, t_aligned_labels = build_data(train_data)
    v_tokenized_inputs, v_aligned_labels = build_data(dev_data)
    tag2idx = generate_tag_ids(train_data)
    # print(tag2idx["DET"])

    train_dataset = POSDataset(t_tokenized_inputs, t_aligned_labels, tag2idx)
    # sample = train_dataset[0]
    # input_ids = sample["input_ids"].to(device)
    # attention_mask = sample["attention_mask"].to(device)
    # labels = sample["labels"].to(device)
    # print(f'Input IDs: {sample["input_ids"]}')
    # print(f'Attention Mask: {sample["attention_mask"]}')
    # print(f'Labels: {sample["labels"]}')
    # print(f'Input IDs Shape: {input_ids.shape}, Attention Mask Shape: {attention_mask.shape}, Labels Shape: {labels.shape}')


    val_dataset = POSDataset(v_tokenized_inputs, v_aligned_labels, tag2idx)
    # sample = val_dataset[0]
    # input_ids = sample["input_ids"].to(device)
    # attention_mask = sample["attention_mask"].to(device)
    # labels = sample["labels"].to(device)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)  # builds the batches for loading data
    # for batch in train_loader:
    #             input_ids = batch["input_ids"].to(device)
    #             attention_mask = batch["attention_mask"].to(device)
    #             labels = batch["labels"].to(device)
    #             print(f'Input IDs Shape: {input_ids.shape}, Attention Mask Shape: {attention_mask.shape}, Labels Shape: {labels.shape}')

    val_loader = DataLoader(val_dataset, batch_size=8, drop_last=True)
    # for batch in val_loader:
    #             input_ids = batch["input_ids"].to(device)
    #             attention_mask = batch["attention_mask"].to(device)
    #             labels = batch["labels"].to(device)
    #             print(f'Input IDs Shape: {input_ids.shape}, Attention Mask Shape: {attention_mask.shape}, Labels Shape: {labels.shape}')

    num_labels = len(tag2idx)
    model =  BertForTokenClassification.from_pretrained("google-bert/bert-base-cased", num_labels=num_labels)
    model.to(device)
        
    train(model, train_loader, val_loader)



if __name__ == "__main__":
    main()
