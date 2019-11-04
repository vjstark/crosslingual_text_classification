## Cross Lingual Classification without translation or retraining

This project aim to create a sentiment classification model to be trained in one language and use it without retraining or translation for a new language

## Dependencies
* Python 2/3 with [NumPy](http://www.numpy.org/)/[SciPy](https://www.scipy.org/)
* [scikit-learn](http://scikit-learn.org/)
* [nltk](https://www.nltk.org/)

## Datasets
Amazon review datasets:
* Book review dataset in [*data/amazon-data*](https://github.com/vjstark/crosslingual_text_classification/tree/master/data/amazon-dataset)
* 2000 for training and 2000 for testing
* Rating used as labels for positve or negative sentiment

You can download the English (en) French (es) and German (de) embeddings this way:
```bash
# English MUSE embeddings
curl -o data/wiki.en.vec https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.en.vec
# French MUSE Wikipedia embeddings
curl -o data/wiki.fr.vec https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.fr.vec
# German MUSE Wikipedia embeddings
curl -o data/wiki.de.vec https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.de.vec
```

## Train and test classifier
This project includes testing all language pair to i.e En-En, En-Fr, En-De ,Fr-Fr, Fr-En, Fr-De, De-De, De-En, De-Fr:

To evaluate the results simply run:
```bash
python crosslingual-classification.py
```
