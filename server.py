import cv2
import numpy as np
from mlchain.base import ServeModel

from flair.data import Sentence
from flair.tokenization import JapaneseTokenizer
from flair.models import TextClassifier

class Model():
    def __init__(self):
        self.tokenizer = JapaneseTokenizer("janome")
        self.classifier = TextClassifier.load('./weight/final-model.pt')
    def predict(self, input:str):
        """
        Resize input to 100 by 100.
        Args:
            image (numpy.ndarray): An input image.
        Returns:
            The image (np.ndarray) at 100 by 100.
        """
        # content = open("data/data1/10000positive.txt", encoding="utf8").readlines()
        # init japanese tokenizer
        
        # make sentence (and tokenize)
        sentence = Sentence(input, use_tokenizer=self.tokenizer)    

        # predict class and print
        self.classifier.predict(sentence)
        result = str(sentence.labels[0])
        return result


# Define model
model = Model()

# Serve model
serve_model = ServeModel(model)

model.predict("どういたしまして")
# Deploy model
if __name__ == '__main__':
    from mlchain.server import FlaskServer
    # Run flask model with upto 12 threads
    FlaskServer(serve_model).run(port=5000, threads=12)