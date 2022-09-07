import nltk
from collections import Counter
import heapq
import re
from nltk.corpus import stopwords
from nltk.stem import  WordNetLemmatizer
from transformers import pipeline

class Preprocess():
  def __init__(self):
    pass
  def toLower(self, x):
    '''Converts string to lowercase'''
    return x.lower()
  
  def sentenceTokenize(self, x):
    '''Tokenizes document into sentences'''
    sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    sentences = sent_tokenizer.tokenize(x)
    return sentences
  
  def preprocess_sentences(self, all_sentences):
    '''Tokenizes sentences into words, removes punctuations, stopwords and 
    performs tokenization'''
    word_tokenizer = nltk.RegexpTokenizer(r"\w+")
    sentences = []
    special_characters = re.compile("[^A-Za-z0-9 ]")
    for s in all_sentences:
      # remove punctuation
      s = re.sub(special_characters, " ", s)
      # Word tokenize
      words = word_tokenizer.tokenize(s)
      # Remove Stopwords
      words = self.removeStopwords(words)
      # Perform lemmatization
      words = self.wordnet_lemmatize(words)
      sentences.append(words)
    return sentences

  def removeStopwords(self, sentence):
    '''Removes stopwords from a sentence'''
    stop_words = stopwords.words('english')
    tokens = [token for token in sentence if token not in stop_words]
    return tokens

  def wordnet_lemmatize(self, sentence):
    '''Lemmatizes tokens in a sentence'''
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token, pos='v') for token in sentence]
    return tokens
  
  def complete_preprocess(self, text):
    '''Performs complete preprocessing on document'''
    #Convert text to lowercase
    text_lower = self.toLower(text)
    #Sentence tokenize the document
    sentences = self.sentenceTokenize(text_lower)
    #Preprocess all sentences
    preprocessed_sentences = self.preprocess_sentences(sentences)
    return preprocessed_sentences

  def generate_wordcloud(self, text):
    word_cloud = WordCloud(collocations = False, background_color = 'white').generate(text)
    plt.figure(figsize=(15,8))
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
  
  def calculate_length(self, df):
    df["article_len"] = df["article"].apply(lambda x: len(x.split()))
    df["highlights_len"] = df["highlights"].apply(lambda x: len(x.split()))
    return df
  
  def most_similar_words(self, model, words):
    '''Returns most similar words to a list of words'''
    for word in words:
      print("Most similar to ", word,": ", model.wv.most_similar(word))
  
  def word2vec_model(self, sentences,num_feature, min_word_count, 
                    window_size, down_sampling,  sg):
    '''Creates and trains Word2Vec model'''
    num_thread = 5
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = word2vec.Word2Vec(sentences, 
                          #iter = iteration,
                          vector_size=num_feature, 
                          min_count = min_word_count, 
                          window = window_size, 
                          sample = down_sampling, 
                          workers=num_thread,
                          sg = sg,
                          epochs = 20)
    return model

  def glove_model(self, sentences, window_size, num_features, lr, iterations):
    '''Creates and trains GloVe model'''
    num_thread = 5
    corpus = Corpus() 
    # Create word co occurence matrix 
    corpus.fit(sentences, window=window_size)
    glove = Glove(no_components=num_features, learning_rate=lr)
    # Fit model
    glove.fit(corpus.matrix, epochs=iterations, no_threads=num_thread)
    glove.add_dictionary(corpus.dictionary)
    return glove
  
  def most_similar_words_glove(self, model, words):
    '''Returns most similar words to a list of words for GloVe model'''
    for word in words:
      print("Most similar to ", word,": ", model.most_similar(word))
  
  def top_10_frequent_words(self, model):
    '''Returns top 10 frequent words'''
    # sort model vocab according to top frequent words
    model.sorted_vocab
    top_words = model.wv.index_to_key[:10]
    return top_words


class NewsSummarization():
  def __init__(self):
    pass
  def extractive_summary(self, text, sentence_len = 8, num_sentences = 3):
    '''Generates extractive summary of num_sentences length using sentence scoring'''
    word_frequencies = {}
    # Instantiate Custom Preprocessor class
    preprocessor = Preprocess()
    # preprocess and tokenize article
    tokenized_article = preprocessor.complete_preprocess(text)
    #calculate word frequencies
    for sentence in tokenized_article:
      for word in sentence:
        if word not in word_frequencies.keys():
          word_frequencies[word] = 1
        else:
          word_frequencies[word] += 1
    #get maximum frequency for score normalisation
    maximum_frequency = max(word_frequencies.values())
    #normalize word frequency
    for word in word_frequencies.keys():
          word_frequencies[word] = (word_frequencies[word]/maximum_frequency)
    sentence_scores = {}

    # score sentences by adding word scores
    sentence_list = nltk.sent_tokenize(text)
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) > sentence_len:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]
    # get sentences with largest sentence scores
    summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    # join and get extractive summary
    summary = ' '.join(summary_sentences)
    return summary

  def get_rouge_score(self, actual_summary, generated_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(actual_summary, generated_summary)
    return scores

  def evaluate_extractive(self, dataset, metric):
    summaries = [self.extractive_summary(text) for text in dataset["article"]]
    score =  metric.compute(predictions=summaries, references=dataset["highlights"])
    rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    rouge_dict = dict((rn, round(score[rn].mid.fmeasure * 100, 2)) for rn in rouge_names)
    return rouge_dict

  def evaluate_abstractive(self, dataset, metric, summarizer):
    summaries =  [summarizer(text, max_length = 120, min_length = 80, do_sample = False)[0]['summary_text'] for text in dataset["article"]]
    score =  metric.compute(predictions=summaries, references=dataset["highlights"])
    rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    rouge_dict = dict((rn, round(score[rn].mid.fmeasure * 100, 2)) for rn in rouge_names)
    return rouge_dict
