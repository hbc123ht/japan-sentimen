import cv2
import numpy as np
from mlchain.base import ServeModel

from flair.data import Sentence
from flair.tokenization import JapaneseTokenizer
from flair.models import TextClassifier

class Model():

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
        tokenizer = JapaneseTokenizer("janome")
        # make sentence (and tokenize)
        print(input)
        sentence = Sentence(input, use_tokenizer=tokenizer)    

        classifier = TextClassifier.load('./weight/final-model.pt')
        # predict class and print
        classifier.predict(sentence)
        print(sentence.labels)
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