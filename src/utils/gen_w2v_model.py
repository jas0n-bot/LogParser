"""Script to generate w2v model from 'test.txt' file."""
import sys
from gensim.models import word2vec


def main():
    """Main entry"""
    g_vec_size = 30

    if len(sys.argv) == 2:
        path = sys.argv[1]
    else:
        path = ''

    sentence = word2vec.LineSentence(path + 'test.txt')
    w2v = word2vec.Word2Vec(sentence, size=g_vec_size, min_count=3, workers=16)
    w2v.save(path + 'w2v.model')


main()
