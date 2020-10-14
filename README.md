# Nick.Udic.Word2Vec

依賴套件：
- gensim：訓練詞向量
- opencc：簡、繁轉換
- jieba：分詞

```
pip3 install gensim
pip3 install jieba
pip3 install opencc
```

## 以 gensim 訓練中文詞向量

> 訓練 Work2Vec，流程為：cwiki -> segment -> train

### cwiki 解析wiki資料
```
python3 main.py cwiki -i ./wiki/zhwiki-20201001-pages-articles.xml.bz2 -o ./output/wiki_20201001_texts.txt
```

### segment 進行分詞
```
python3 main.py segment -i ./output/wiki_20201001_texts.txt -o ./output/wiki_20201001_seq.txt
```

### train 進行詞向量解析
```
python3 main.py train -i ./output/wiki_20201001_seq.txt -o ./output/wiki_20201001_word2vec.model
```

### query 進行關鍵字查詢
```
python3 main.py query -i ./output/wiki_word2vec.model -k 蘋果
```

## 以 gensim 訓練情緒模型並判斷句子情緒

> 流程為： train 情緒 Model -> querynp


### train 訓練正、負向情緒模型
調整資料集：
- `./data_set/neg_cus.txt`
- `./data_set/pos_cus.txt`

進行訓練
```
python3 main.py train -i ./data_set/neg_cus.txt -o ./output/neg_cus_word2vec.model
python3 main.py train -i ./data_set/neg_cus.txt -o ./output/neg_cus_word2vec.model
```

### querynp 進行情緒判斷
參數
- ip：正向情緒模型路徑
- in：負向情緒模型路徑
- k：查詢語句

```
python3 main.py querynp -ip ./output/pos_cus_word2vec.model -in ./output/neg_cus_word2vec.model -k 今天心情很好
```