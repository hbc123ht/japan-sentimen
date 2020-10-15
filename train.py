from flair.data import Sentence
from flair.tokenization import JapaneseTokenizer

# init japanese tokenizer
tokenizer = JapaneseTokenizer("janome")

from torch.optim.lr_scheduler import OneCycleLR
from flair.data import Corpus
from flair.datasets import ClassificationCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentLSTMEmbeddings, CharacterEmbeddings, BytePairEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from pathlib import Path

# load corpus containing training, test and dev data and if CSV has a header, you can skip it
corpus: Corpus = ClassificationCorpus(Path('./data1/train/'),
                                      tokenizer=tokenizer
                                      )
word_embeddings = [
    BytePairEmbeddings('en', dim=50),
    BytePairEmbeddings('ja', dim=100),
    FlairEmbeddings('ja-forward'), 
    FlairEmbeddings('ja-backward')
]

document_embeddings = DocumentLSTMEmbeddings(word_embeddings, hidden_size=512, reproject_words=True, reproject_words_dimension=256)
classifier = TextClassifier(document_embeddings, label_dictionary=corpus.make_label_dictionary(), multi_label=False)

# train as always
trainer = ModelTrainer(classifier, corpus)

#load checkpoint
checkpoint = './onecycle_ner/checkpoint.pt'
trainer = ModelTrainer.load_checkpoint(checkpoint, corpus)

# set one cycle LR as scheduler
trainer.train('onecycle_ner',
              scheduler=OneCycleLR,
              mini_batch_size = 32,
              checkpoint = True,
              max_epochs=40)