
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import io
import re
import string
# import unidecode
from string import punctuation
import nltk
nltk.download('punkt')
from gensim.parsing.preprocessing import STOPWORDS
import emoji
from emoji import UNICODE_EMOJI
import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from wordcloud import WordCloud
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer


gensim_stop_words = STOPWORDS.difference(set(['not']))

def extract_emoji(text):
    em = set()
    em_annotate = []
    count=0
    emojis_iter = map(lambda y: y, emoji.UNICODE_EMOJI['en'].keys())
    regex_set = re.compile('|'.join(re.escape(em) for em in emojis_iter))
    emoj = regex_set.findall(text)
    em = em.union(set(emoj))
    em_annotate += [UNICODE_EMOJI['en'][e].upper()[1:-1] for e in em]
    count += len(emoj)
#     print(f"Total emojis: {count} | unique emojis: {len(em)}")
    return list(em), list(em_annotate)

def remove_emoji(text):
    em, em_annotate = extract_emoji(text)
    for e, label in zip(em, em_annotate):
        text = text.replace(str(e), " "+label+"")
    return text

def remove_spaces_and_new_lines(text):
    Formatted_text = text.replace('\\n', ' ').replace('\n', ' ').replace('\t',' ').replace('\\', ' ').replace('. com', '.com')
    pattern = re.compile(r'\s+')
    Without_whitespace = re.sub(pattern, ' ', Formatted_text)
    Formatted_text = Without_whitespace.replace('?', ' ? ').replace(')', ') ')
    return Formatted_text

def replace_number(text):
    text = re.sub(r" [0-9]+ ", " NN ", text)
    text = re.sub(r'([a-z]*)[0-9]+([a-z]+)|([a-z]+)[0-9]+([a-z]*)', r'\1\2', text)
    return text

def annotate_links(text):
    link_pattern = re.sub(r'http\S+', '', text)
    annotated = re.sub(r"\ [A-Za-z]*\.com", "", link_pattern)
    return annotated

def remove_special_chars(text):
    text = re.sub(r"[^A-Za-z ]+", '', text)
    return text.translate(str.maketrans('', '', punctuation))

CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have",
"ur":"your",
"r":"are",
"u":"you",
}
def expand_contractions(text, contraction_mapping =  CONTRACTION_MAP):
    token_list = text.lower().split(' ')
    for word in token_list:
         if word in CONTRACTION_MAP:
                token_list = [item.replace(word, CONTRACTION_MAP[word]) for item in token_list]

    combined = ' '.join(str(e) for e in token_list)
    return combined

def preprocess_text(text_list):
    cleaned_text = []
    for text in tqdm.tqdm(text_list):
        text = text.lower()
        text = expand_contractions(text)
        text = replace_number(text)
        text = remove_emoji(text)
        text = annotate_links(text)
        text = remove_special_chars(text)
        text = remove_spaces_and_new_lines(text)
        cleaned_text.append(text)

    return cleaned_text
def get_word_freq(cv, text_list):
  words = cv.fit_transform(text_list)

  sum_words = words.sum(axis=0)

  words_freq = [(word, sum_words[0, i]) for word, i in cv.vocabulary_.items()]
  words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
  return words_freq

def draw_word_cloud(words_freq, fname, title="WordCloud - Vocabulary from Reviews"):
  wordcloud = WordCloud(background_color = 'white', width = 1000, height = 1000).generate_from_frequencies(dict(words_freq))

  plt.figure(figsize=(10,8))
  plt.imshow(wordcloud)
  plt.title(title, fontsize = 22)
  plt.savefig(fname)

def run(df,id):
    df = df.fillna("")
    df['text'] = df['text'].apply(lambda x: re.sub('^RT @[A-Za-z0-9]+:', '', x))
    df['text'] = preprocess_text(df['text'])
    cv = CountVectorizer(stop_words = 'english')
    words = cv.fit_transform(df['text'])


    sum_words = words.sum(axis=0)


    words_freq = [(word, sum_words[0, i]) for word, i in cv.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)

    frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])
    frequency.head(30).plot(x='word', y='freq', kind='bar', figsize=(10, 8), color = 'blue')

    plt.title("Most Frequently Occuring Words - Top 30")
    # plt.savefig("output_img/frequency.jpg")
    plt.savefig("D:/Sentiment-Front/sentiment-analysis/src/assets/frequency.jpg")



    # return dict(df[['text']])
    # tokenizer = pickle.load(open('tokenizer.pkl','rb'))

    # test_encoded_text, test_max_len = encode_text(tokenizer, df['text'].tolist())
    # test_encoded_text = pad_data(test_encoded_text,47)
    # test_encoded_text = pad_data(test_encoded_text,62)
    # X_test = np.array(df['text'].tolist())
    # X_test_1 = Tfidf_vect.fit_transform(X_test).toarray()
    X_test = np.array(df['text'].tolist())
    Tfidf_vect = TfidfVectorizer(max_features=62)
    X_test_1 = Tfidf_vect.fit_transform(X_test).toarray()


    model = pickle.load(open('model1.pkl', 'rb'))

    predictions = model.predict(X_test_1)

    i = {0:'Sports',1:'Development',2:'Education',3:'Others'}
    df['output']=list(map(lambda x:i[x],predictions.tolist()))


    # df['polarity'] = df['text'].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
    # df['subjectivity'] = df['text'].apply(lambda tweet: TextBlob(tweet).sentiment.subjectivity)

    # neutral = df.loc[df['polarity'].apply(lambda x: True if int(x)==0 else False), "text"].tolist()
    # positive = df.loc[df['polarity'].apply(lambda x: True if x>0 else False), "text"].tolist()
    # negative = df.loc[df['polarity'].apply(lambda x: True if x<0 else False), "text"].tolist()

    sports = df.loc[df['output'].apply(lambda x: True if x=='Sports' else False), "text"].tolist()
    development = df.loc[df['output'].apply(lambda x: True if x=='Development' else False), "text"].tolist()
    education = df.loc[df['output'].apply(lambda x: True if x=='Education' else False), "text"].tolist()
    others = df.loc[df['output'].apply(lambda x: True if x=='Others' else False), "text"].tolist()



    sports_words = get_word_freq(cv, sports)
    development_words = get_word_freq(cv, development)
    education_words = get_word_freq(cv, education)
    others_words = get_word_freq(cv, others)



    draw_word_cloud(sports_words, "D:/Sentiment-Front/sentiment-analysis/src/assets/sports.jpg", "WordCloud - For Sports related Tweets")
    draw_word_cloud(development_words, "D:/Sentiment-Front/sentiment-analysis/src/assets/development.jpg", "WordCloud - For Development related Tweets")
    draw_word_cloud(education_words, "D:/Sentiment-Front/sentiment-analysis/src/assets/education.jpg", "WordCloud - For Education related Tweets")
    draw_word_cloud(others_words, "D:/Sentiment-Front/sentiment-analysis/src/assets/others.jpg", "WordCloud - For Others related Tweets")



    #Creating PieChart
    counts = dict(df['output'].value_counts())
    plt.pie(counts.values(),labels=counts.keys(),autopct='%1.1f%%' )
    plt.title("Pie Chart for Categories")
    plt.savefig("D:/Sentiment-Front/sentiment-analysis/src/assets/piecart.jpg")


    # return dict(df[['text','output']])
    return {'text':df['text'].tolist(),'output':df['output'].tolist()}

def encode_text(tokenizer, text):
  encoded_text = tokenizer.texts_to_sequences(text)
  print(type(encoded_text), len(encoded_text), '\n', encoded_text[0:5])

  # Getting the Average, standard devaition & Max length of Encoded Training Data
  texts_len = [len(x) for x in encoded_text]
  max_len = max(texts_len)
  print ("Largest string of text is: :", max_len)
  print ("AVG length is :", mean(texts_len))
  print('Std dev is:', np.std(texts_len))
  print('mean+ sd.deviation value for encoded text is:', '\n', int(mean(texts_len)) + int(np.std(texts_len)))

  return encoded_text, max_len

def pad_data(encoded_text, max_len):
    padded_seq = pad_sequences(encoded_text, maxlen=max_len, padding="post")
    print("Shape of padded data is:", padded_seq.shape, type(padded_seq), len(padded_seq),'\n', padded_seq[0:1])
    return padded_seq

if __name__ == '__main__':
    run()
