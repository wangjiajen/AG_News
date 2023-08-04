# <font color="#0000AA"> **基於Transformer架構的新聞文章文本分類**</font>
<br>`本專案旨在展示如何使用Transformer架構進行文本分類，尤其是將新聞文章分類為不同的主題類別。我們使用AG News數據集來建立和評估這個模型。`<br>

## <font color="#CC6600"> **Data Preparation**</font><br>
### 1.安裝套件<br>
```sh
# Clone
git clone https://github.com/wangjiajen/AG_News.git

# Install dependencies.
pipenv install
```
### 2.準備數據<br>
```sh
# 下載AG News數據集，或提供包含新聞文章及其對應標籤的訓練和測試CSV檔案
traindata = pd.read_csv('/home/apple/AG_News/data/train.csv')
testdata = pd.read_csv('/home/apple/AG_News/data/test.csv')
```
## <font color="#CC6600"> **Code Explanation**</font><br>
### 1.資料前處理<br>
```sh
# 將新聞標題和描述結合成一個"summary"欄位
def combine_title_and_description(df):
    df['summary'] = df[['Title', 'Description']].agg('. '.join, axis=1)
    df = df.drop(['Title', 'Description'], axis=1)
    return df

# 使用Tokenizer對文本進行標記化，並對文本序列進行填充，以確保相同長度的輸入
X_data = keras.preprocessing.sequence.pad_sequences(X_data, maxlen=maxlen)
x_testdata = keras.preprocessing.sequence.pad_sequences(x_testdata,maxlen=maxlen)
```
### 2.模型架構<br>
```sh
# 定義的TransformerBlock和TokenAndPositionEmbedding層
class TransformerBlock(layers.Layer):

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = layers.MultiHeadAttention(num_heads=num_heads,
                                             key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        return ({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate
        })


class TokenAndPositionEmbedding(layers.Layer):

    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__()
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(input_dim=vocab_size,
                                          output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config()
        return ({
            'maxlen': self.maxlen,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
        })

inputs = layers.Input(shape=(maxlen, ))
embedding_layer = TokenAndPositionEmbedding(maxlen, max_words, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(4, activation="softmax")(x)
```
### 3.訓練和評估<br>
```sh
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
EPOCHS = 10
checkpoint = tf.keras.callbacks.ModelCheckpoint('transformer_agnews.best.hdf5',
                                                monitor='val_loss',
                                                mode='min',
                                                verbose=1,
                                                save_best_only=True)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
model.fit(X_train,
          y_train,
          batch_size=128,
          epochs=EPOCHS,
          validation_data=(X_val, y_val),
          callbacks=[checkpoint, early_stopping]
         )
```
### 4.預測和分類指標<br>
```sh
# 生成混淆矩陣以及準確度、召回率、精確度和F1分數等分類指標，以評估模型性能
preds = [np.argmax(i) for i in prediction]
cm = confusion_matrix(y_test, preds)
plt.figure()
plt.rcParams.update({'font.size': 30})
plot_confusion_matrix(cm, figsize=(16, 12), hide_ticks=True, cmap=plt.cm.Blues)
plt.xticks(range(4), labels, fontsize=15)
plt.yticks(range(4), labels, fontsize=15)
plt.show()
plt.savefig('test.png', bbox_inches="tight")
y_pred = np.argmax(prediction, axis=1)

print(classification_report(y_test, y_pred, labels=[0, 1, 2, 3]))
```
## <font color="#CC6600">**參考資料**</font>
<br>[1]:https://keras.io/examples/nlp/text_classification_from_scratch/<br>
[2]:https://www.kaggle.com/code/jannesklaas/17-nlp-and-word-embeddings/notebook<br>
[3]:https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset<br>
