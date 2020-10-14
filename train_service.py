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

    # # jieba custom setting.
    jieba.set_dictionary('jieba_dict/dict.txt.big')

    output = open(output_name, 'w', encoding='utf-8')
    with open(file_name, 'r', encoding='utf-8') as content:
        for texts_num, line in enumerate(content):
            line = line.strip('\n')
            words = jieba_cut(line)
            for word in words:
                if word == "\n":
                    output.write("\n")
                else:
                    output.write(word + ' ')

            if (texts_num + 1) % 10000 == 0:
                logging.info("已完成前 %d 行的斷詞" % (texts_num + 1))
    output.close()
    logging.info("結束 segment")


def jieba_cut(line, use_stopwords=True, is_EOL=True):
    # load stopwords set
    stopword_set = set()
    if use_stopwords:
        with open('jieba_dict/stopwords.txt', 'r', encoding='utf-8') as stopwords:
            for stopword in stopwords:
                stopword_set.add(stopword.strip('\n'))

    result = []
    words = jieba.cut(line, cut_all=False)
    for w in words:
        if w not in stopword_set:
            result.append(w)

    if is_EOL:
        result.append('\n')
    return result


def train(file_name, output_name):
    logging.info("開始 train")
    sentences = word2vec.LineSentence(file_name)
    # model = word2vec.Word2Vec(sentences, size=200,min_count=10)
    model = word2vec.Word2Vec(sentences, size=10, min_count=1)

    # 保存模型
    model.save(output_name)

    logging.info("結束 train")


def query(file_name, keyword, top=100):
    model = models.Word2Vec.load(file_name)

    q_list = keyword.split(',')

    if len(q_list) == 1:
        print("【" + keyword + "】:相似詞前 " + str(top) + " 排序")
        if keyword in model:
            print(model[keyword].sum() / len(model[keyword]))
            res = model.most_similar(keyword, topn=int(top))
            for item in res:
                print(item[0] + "," + str(item[1]))

    if len(q_list) == 2:
        print("計算 Cosine 相似度")
        res = model.similarity(q_list[0], q_list[1])
        print(res)

    if len(q_list) == 3:
        print("%s之於%s，如%s之於" % (q_list[0], q_list[1], q_list[2]))
        res = model.most_similar([q_list[0], q_list[2]], [q_list[1]], topn=100)
        for item in res:
            print(item[0] + "," + str(item[1]))


def querynp(file_name_pos, file_name_neg, keyword):
    model_pos = models.Word2Vec.load(file_name_pos)
    model_neg = models.Word2Vec.load(file_name_neg)

    inverse_set = set()
    with open('jieba_dict/inversewords.txt', 'r', encoding='utf-8') as inversewords:
        for inverseword in inversewords:
            inverse_set.add(inverseword.strip('\n'))

    # jieba custom setting.
    jieba.set_dictionary('jieba_dict/dict.txt.big')
    words = jieba_cut(keyword, False, False)

    t1 = 0
    t2 = 0

    is_inverse = False

    for w in words:
        print(w)
        if w in inverse_set:
            is_inverse = not is_inverse

        if w in model_pos:
            t1 += sum(i[1] for i in model_pos.predict_output_word([w]))

        if w in model_neg:
            t2 += sum(i[1] for i in model_neg.predict_output_word([w]))

    print('正向概率總和為:%.8f' % t1)
    print('反向概率總和為:%.8f' % t2)

    state = not (t1 > t2) if is_inverse else t1 > t2
    print("正面" if state else "負面")
