# Nick.Udic.Word2Vec
以 gensim 訓練中文詞向量

> 訓練 Work2Vec，流程為：cwiki -> segment -> train

## cwiki 解析wiki資料
```
python3 main.py cwiki -i ./wiki/zhwiki-20201001-pages-articles.xml.bz2 -o ./output/wiki_20201001_texts.txt
```

## segment 進行分詞
```
python3 main.py segment -i ./output/wiki_20201001_texts.txt -o ./output/wiki_20201001_seq.txt
```

## train 進行詞向量解析
```
python3 main.py train -i ./output/wiki_20201001_seq.txt -o ./output/wiki_20201001_word2vec.model
```

## query 進行關鍵字查詢
```
python3 main.py query -i ./output/wiki_word2vec.model -k 蘋果
```