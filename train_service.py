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
    #
    # # load stopwords set
    # stopword_set = set()
    # with open('jieba_dict/stopwords.txt', 'r', encoding='utf-8') as stopwords:
    #     for stopword in stopwords:
    #         stopword_set.add(stopword.strip('\n'))

    output = open(output_name, 'w', encoding='utf-8')
    with open(file_name, 'r', encoding='utf-8') as content:
        for texts_num, line in enumerate(content):
            line = line.strip('\n')
            words = jieba_cut(line)
            for word in words:
                output.write(word + ' ')

            if (texts_num + 1) % 10000 == 0:
                logging.info("已完成前 %d 行的斷詞" % (texts_num + 1))
    output.close()
    logging.info("結束 segment")


def jieba_cut(line):
    # load stopwords set
    stopword_set = set()
    with open('jieba_dict/stopwords.txt', 'r', encoding='utf-8') as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip('\n'))

    result = []
    words = jieba.cut(line, cut_all=False)
    for w in words:

        if w not in stopword_set:
            result.append(w)

        # result.append('\n')
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


def query2(file_name_pos, file_name_neg, keyword):
    model_pos = models.Word2Vec.load(file_name_pos)
    model_neg = models.Word2Vec.load(file_name_neg)

    # jieba custom setting.
    jieba.set_dictionary('jieba_dict/dict.txt.big')

    # words = jieba.cut(keyword, cut_all=False)
    words = jieba_cut(keyword)
    total_pos = 0
    total_neg = 0

    t_p = 0
    t_n = 0

    t1 = 0
    t2 = 0

    # a1=model_pos.n_similarity(words, model_pos.wv.index2word)
    # a2=model_neg.n_similarity(words, model_neg.wv.index2word)
    # res_pos = 0
    # res_neg = 0
    # for i in range(len(words)):
    #     for w1 in words:
    #         if words[i] == w1:
    #             continue
    #         print(words[i], w1)
    #         if words[i] in model_pos and w1 in model_pos:
    #             res_pos += model_pos.similarity(words[i], w1)
    #             print("res_pos",res_pos)
    #         if words[i] in model_neg and w1 in model_neg:
    #             res_neg += model_neg.similarity(words[i], w1)
    #             print("res_neg",res_neg)

    # for i in range(len(words)-1):
    #     if words[0] == words[i+1]:
    #         continue
    #     print(words[0], words[i + 1])
    #     if words[0] in model_pos and words[i + 1] in model_pos:
    #         res_pos += model_pos.similarity(words[0], words[i + 1])
    #         print("res_pos", res_pos)
    #     if words[0] in model_neg and words[i + 1] in model_neg:
    #         res_neg += model_neg.similarity(words[0], words[i + 1])
    #         print("res_neg", res_neg)

    for w in words:
        print(w)
        if w in model_pos:
            # print(model_pos[w])
            score1 = model_pos[w].sum() / len(model_pos[w])

            # print(model_pos.predict_output_word([w]))
            t1 += sum(i[1] for i in model_pos.predict_output_word([w]))

            print(w, "正向分數:", score1)
            total_pos += score1
            t_p = t_p + 1

        if w in model_neg:
            # print(model_neg[w])
            # print(model_neg.wv.similar_by_vector(w))
            score2 = model_neg[w].sum() / len(model_neg[w])

            t2 += sum(i[1] for i in model_neg.predict_output_word([w]))

            print(w, "反向分數:", score2)
            total_neg += score2
            t_n = t_n + 1

    print("total_pos = ", total_pos)
    print("total_neg = ", total_neg)
    #
    # print("t_p = ", t_p)
    # print("t_n = ", t_n)

    print('正向概率總和為:%.8f' % t1)
    print('反向概率總和為:%.8f' % t2)

    # print("res_pos = ", res_pos)
    # print("res_neg = ", res_neg)

    # print(total_pos > total_neg)
    # print(res_pos > res_neg)
    print(t1 > t2)
