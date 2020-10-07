import logging

import jieba
import opencc
from gensim import models
from gensim.corpora import WikiCorpus
from gensim.models import word2vec

def wiki_to_txt(file_name, output_name):

    logging.info("開始 wiki_to_txt")
    wiki_corpus = WikiCorpus(file_name, dictionary={})
    texts_num = 0
    converter = opencc.OpenCC('s2t.json')
    with open(output_name, 'w', encoding='utf-8') as output:
        for texts in wiki_corpus.get_texts():
            r = converter.convert(' '.join(texts))
            output.write(r + '\n')
            texts_num += 1
            if texts_num % 10000 == 0:
                logging.info("已處理 %d 篇文章" % texts_num)

    logging.info("結束 wiki_to_txt")


def segment(file_name, output_name):
    logging.info("開始 segment")

    # jieba custom setting.
    jieba.set_dictionary('jieba_dict/dict.txt.big')

    # load stopwords set
    stopword_set = set()
    with open('jieba_dict/stopwords.txt', 'r', encoding='utf-8') as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip('\n'))

    output = open(output_name, 'w', encoding='utf-8')
    with open(file_name, 'r', encoding='utf-8') as content:
        for texts_num, line in enumerate(content):
            line = line.strip('\n')
            words = jieba.cut(line, cut_all=False)
            for word in words:
                if word not in stopword_set:
                    output.write(word + ' ')
            output.write('\n')

            if (texts_num + 1) % 10000 == 0:
                logging.info("已完成前 %d 行的斷詞" % (texts_num + 1))
    output.close()
    logging.info("結束 segment")


def train(file_name, output_name):
    logging.info("開始 train")
    sentences = word2vec.LineSentence(file_name)
    model = word2vec.Word2Vec(sentences, size=250)

    # 保存模型
    model.save(output_name)

    logging.info("結束 train")


def query(file_name, keyword, top=100):
    model = models.Word2Vec.load(file_name)
    print("【"+keyword+"】:相似詞前 "+str(top)+" 排序")
    res = model.most_similar(keyword, topn=top)
    for item in res:
        print(item[0] + "," + str(item[1]))