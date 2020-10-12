import jieba

def main():

    jieba.set_dictionary('./jieba_dict/dict.txt.big')

    # load stopwords set
    stopword_set = set()
    with open('jieba_dict/stopwords.txt', 'r', encoding='utf-8') as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip('\n'))

    input = "我喜歡吃香蕉"
    words = jieba.cut(input, cut_all=False)
    for word in words:
        if word not in stopword_set:
            print(word + ' ')


if __name__ == '__main__':
    main()