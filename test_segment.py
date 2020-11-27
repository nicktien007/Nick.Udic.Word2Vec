import jieba

def main():

    jieba.set_dictionary('./jieba_dict/dict.txt.big')

    # userdict 範例：
    # 三師父 70 nr
    # 三師父 70 nr
    # 載入自定義詞典
    # jieba.load_userdict("userdict.txt")

    # load stopwords set
    stopword_set = set()
    with open('jieba_dict/stopwords.txt', 'r', encoding='utf-8') as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip('\n'))

    input = "明末清初著名軍事將領，曾因「引清兵入關」而被世人斥責為漢奸，他的名字叫做"
    words = jieba.cut(input, cut_all=False)
    for word in words:
        if word not in stopword_set:
            print(word + ' ')


if __name__ == '__main__':
    main()