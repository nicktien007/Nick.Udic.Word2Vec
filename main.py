import logging

import jieba
import opencc
from gensim import models
from gensim.corpora import WikiCorpus
from gensim.models import word2vec


def wiki_to_txt():
    logging.info("開始 wiki_to_txt")
    wiki_corpus = WikiCorpus("./wiki/zhwiki-20200920-pages-articles-1.xml.bz2", dictionary={})
    texts_num = 0
    converter = opencc.OpenCC('s2t.json')
    with open("output/wiki_texts.txt", 'w', encoding='utf-8') as output:
        for texts in wiki_corpus.get_texts():
            r = converter.convert(' '.join(texts))
            output.write(r + '\n')
            texts_num += 1
            if texts_num % 10000 == 0:
                logging.info("已處理 %d 篇文章" % texts_num)

    logging.info("結束 wiki_to_txt")


def segment():
    logging.info("開始 segment")

    # jieba custom setting.
    jieba.set_dictionary('jieba_dict/dict.txt.big')

    # load stopwords set
    stopword_set = set()
    with open('jieba_dict/stopwords.txt', 'r', encoding='utf-8') as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip('\n'))

    output = open('output/wiki_seg.txt', 'w', encoding='utf-8')
    with open('./output/wiki_texts.txt', 'r', encoding='utf-8') as content:
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


def train():
    logging.info("開始 train")
    sentences = word2vec.LineSentence("./output/wiki_seg.txt")
    model = word2vec.Word2Vec(sentences, size=250)

    # 保存模型，供日後使用
    model.save("./output/wiki_word2vec.model")

    logging.info("結束 train")


def query():
    model = models.Word2Vec.load('./output/wiki_word2vec.model')
    q = "佐助"
    print("相似詞前 100 排序")
    res = model.most_similar(q, topn=100)
    for item in res:
        print(item[0] + "," + str(item[1]))


def main():
    Init_logging()
    # wiki_to_txt()
    # segment()
    # train()
    query()


def Init_logging():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


if __name__ == '__main__':
    main()
