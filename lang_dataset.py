from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel

class LanguageDataset():
    def __init__(self, tokeniser_name, dataset_name):
        tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        #model = AutoModel.from_pretrained("prajjwal1/bert-tiny")


if __name__ == "__main__":
    LanguageDataset("bla", "bla")