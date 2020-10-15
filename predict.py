from flair.data import Sentence
from flair.tokenization import JapaneseTokenizer
from flair.models import TextClassifier

# init japanese tokenizer
tokenizer = JapaneseTokenizer("janome")
# make sentence (and tokenize)
sentence = Sentence("締め切りが嫌い", use_tokenizer=tokenizer)    

classifier = TextClassifier.load('./onecycle_ner/final-model.pt')
# predict class and print
classifier.predict(sentence)
print(sentence.labels)


