import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader


class SSTDataset(Dataset):
    """
    Stanford Sentiment Treebank V2.0
    Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank
    Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher Manning, Andrew Ng and Christopher Potts
    Conference on Empirical Methods in Natural Language Processing (EMNLP 2013)
    """

    def __init__(self, filename, maxlen, tokenizer, return_sentences=False):
        # Store the contents of the file in a pandas dataframe
        self.df = pd.read_csv(filename, delimiter='\t', header=None)\
                    .rename(columns={0: "label", 1: "sentence"})
        # Initialize the tokenizer for the desired transformer model
        self.tokenizer = tokenizer
        # Maximum length of the tokens list to keep all the sequences of fixed size
        self.maxlen = maxlen
        # whether to tokenize or return raw setences
        self.return_sentences = return_sentences
        # labels
        self.label_mapping = {0: 'negative', 1: 'positive'}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Select the sentence and label at the specified index in the data frame
        sentence = self.df.loc[index, 'sentence']
        label = self.df.loc[index, 'label']
        # Preprocess the text to be suitable for the transformer
        if self.return_sentences:
            return sentence, label
        else:
            input_ids, attention_mask = self.process_sentence(sentence)
            return input_ids, attention_mask, label

    def process_sentence(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        if len(tokens) < self.maxlen:
            tokens = tokens + \
                ['[PAD]' for _ in range(self.maxlen - len(tokens))]
        else:
            tokens = tokens[:self.maxlen-1] + ['[SEP]']
        # Obtain the indices of the tokens in the BERT Vocabulary
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids)
        # Obtain the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attention_mask = (input_ids != 0).long()
        return input_ids, attention_mask


class IndexedTensorDataset(TensorDataset): 
    def __getitem__(self, index): 
        val = super(IndexedTensorDataset, self).__getitem__(index)
        return val + (index,)
    
class IndexedDataset(Dataset): 
    def __init__(self, ds, sample_weight=None): 
        super(Dataset, self).__init__()
        self.dataset = ds
        self.sample_weight=sample_weight
    
    def __getitem__(self, index): 
        val = self.dataset[index]
        if self.sample_weight is None: 
            return val + (index,)
        else: 
            weight = self.sample_weight[index]
            return val + (weight,index)
    def __len__(self): 
        return len(self.dataset)


class NormalizedRepresentation(torch.nn.Module): 
    def __init__(self, loader, metadata, device='cuda', tol=1e-5): 
        super(NormalizedRepresentation, self).__init__()

        assert metadata is not None
        self.device = device
        self.mu = metadata['X']['mean']
        self.sigma = torch.clamp(metadata['X']['std'], tol)

    def forward(self, X): 
        return (X - self.mu.to(self.device))/self.sigma.to(self.device)
    

def add_index_to_dataloader(loader, sample_weight=None): 
    return DataLoader(IndexedDataset(loader.dataset, sample_weight=sample_weight), 
                      batch_size=loader.batch_size, 
                      sampler=loader.sampler, 
                      num_workers=loader.num_workers, 
                      collate_fn=loader.collate_fn, 
                      pin_memory=loader.pin_memory, 
                      drop_last=loader.drop_last, 
                      timeout=loader.timeout, 
                      worker_init_fn=loader.worker_init_fn, 
                      multiprocessing_context=loader.multiprocessing_context
                      )