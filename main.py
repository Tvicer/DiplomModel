from __future__ import annotations

import http

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

import seaborn as sns
import nltk
import requests

from string import punctuation
from collections import Counter, namedtuple
from itertools import product
from typing import Dict, List
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering

from wiki_ru_wordnet import WikiWordnet
from typing import Set

from bs4 import BeautifulSoup

from natasha import Doc, MorphVocab, Segmenter, NewsEmbedding, \
                    NewsMorphTagger, NewsSyntaxParser
from razdel import sentenize, tokenize
from nltk.corpus import stopwords

from natasha import (
    Segmenter, MorphVocab,
    NewsEmbedding, NewsMorphTagger, NewsSyntaxParser,
    Doc
)
import natasha
import ipymarkup
from typing import Any, List, Dict, Tuple, Optional, Set

from sklearn import preprocessing, metrics
from xgboost import XGBClassifier

import lime
import lime.lime_tabular
from sklearn.model_selection import train_test_split
import os
import json
import re
import joblib
import time
import Stemmer
import pymorphy2

from itertools import combinations
from nltk.stem import SnowballStemmer
from enum import Enum

from sqlalchemy import create_engine
import psycopg2 as pg

def tokenize_wrapper(in_str: str):
  """
  Токенизация текста с помощью Наташи
  """
  return [item.text for item in tokenize(in_str) if item.text not in punctuation]


def sentenize_wrapper(in_str: str):
  """
  Разделения текста на предложения с помощью Наташи
  """
  return [sent.text for sent in sentenize(in_str)]


def get_lemms(in_str: str):
  """
  Разделяем полученный текст на токены и получаем лемму для каждого
  """
  curr_doc = Doc(in_str)
  curr_doc.segment(segmenter)
  curr_doc.tag_morph(morph_tagger)
  for token in curr_doc.tokens:
    token.lemmatize(morph_vocab)
  return [token.lemma for token in curr_doc.tokens if token.lemma not in punctuation]


def get_pos(in_str: str):
  """
  Получим части речи для каждого токена в предложении
  """
  curr_doc = Doc(in_str)
  curr_doc.segment(segmenter)
  curr_doc.tag_morph(morph_tagger)
  return [(token.text, token.pos) for token in curr_doc.tokens]

"""# Графики

Вспомогательные функции
"""

def get_statisctics(l, text=None):
  text = '' if text is None else text.strip()+' '

  print(f"MAX {text}- {np.max(l)}")
  print(f"MIN {text}- {np.min(l)}")
  print(f"MEAN {text}- {np.mean(l)}")
  print(f"MEDIAN {text}- {np.median(l)}")



class SentencesPreprocessor():
  """
      Standart sentence normalizer
  """

  def __init__(self, stopword_lang="russian", min_token_size=3):
    self._russian_stopwords = stopwords.words('russian')

    self._min_token_size = min_token_size
    self._inverse_word = ['не', 'нет', 'нету', 'ни']
    self._increase_word = ['очень']

    # self._morph_vocab = MorphVocab()
    # self._morph_analyzer = pymorphy2.MorphAnalyzer()


  def normalize_sentence(self, sentence):
    """
    Standart sentense normalizer

    :param sentence: input sentence
    :return: normalized sentence
    """
    curr_doc = Doc(sentence)
    curr_doc.segment(segmenter)
    curr_doc.tag_morph(morph_tagger)
    for token in curr_doc.tokens:
      token.lemmatize(morph_vocab)
    return [token.lemma.replace('ё', 'е') for token in curr_doc.tokens if token.lemma not in punctuation \
              and token.lemma.strip().isalpha() and len(token.lemma) >= self._min_token_size]

  def normalize_iter(self, iter):
    l = []
    for word in iter:
      prep = self.normalize_sentence(word)
      if prep:
        l.append(prep[0])
    return l

  def transform(self, df: pd.DataFrame, col_sentence='tell'):
    """
    Using Natasha make preprocessing operation for your dataframe of sentences

    :param df:
    :param col_sentence:
    :return:
    """
    df[col_sentence] = df[col_sentence].apply(self.normalize_sentence_via_mystem)
    return df


def get_word_uniq(text, prep=True):
  """
  На вход получаем текст, а возвращаем словарь с подсчетом уникальных токенов
  """
  return Counter(preprocess(text) if prep else tokenize_wrapper(text))

def get_word_uniq_lst(lst_texts, prep=True):
  """
  На вход получаем список текстов, а возвращаем словарь с подсчетом уникальных токенов
  """
  all_word_lst = []
  preprocessor = SentencesPreprocessor()
  for text in lst_texts:
    all_word_lst.extend(preprocess(text) if prep else tokenize_wrapper(text))
  return Counter(all_word_lst)

class Synonymizer():
  def __init__(self):
    self.wikiwordnet = WikiWordnet()

  def get_synset_word(self, synset):
    return [w.lemma() for w in synset.get_words()]

  def get_synset(self, word):
    synsets = self.wikiwordnet.get_synsets(word)
    return synsets[0] if synsets else None

  def get_hyponyms(self, synset1):
    if synset1 is None:
      return []

    hyponyms = []
    for hyponym in self.wikiwordnet.get_hyponyms(synset1):
      for w in hyponym.get_words():
        hyponyms.append(self.get_synset(w.lemma()))

    return hyponyms

  def get_hypernyms(self, synset1):
    if synset1 is None:
      return []

    hypernyms = []
    for hypernym in self.wikiwordnet.get_hypernyms(synset1):
      for w in hypernym.get_words():
        hypernyms.append(self.get_synset(w.lemma()))

    return hypernyms

  def get_synonyms(self, word) -> Set:
    words = SentencesPreprocessor().normalize_sentence(word)
    if len(words) == 0:
      return set()

    word = words[0]
    synsets = set()
    synset = self.get_synset(word)
    if synset is not None:
      synsets.add(synset)

    for hyponym1 in self.get_hyponyms(self.get_synset(word)):
      synsets.add(hyponym1)
      for hyponym2 in self.get_hyponyms(hyponym1):
        synsets.add(hyponym2)
        for hyponym3 in self.get_hyponyms(hyponym2):
          synsets.add(hyponym3)

    for hypernym1 in self.get_hypernyms(self.get_synset(word)):
      synsets.add(hypernym1)
      for hypernym2 in self.get_hypernyms(hypernym1):
        synsets.add(hypernym2)
        for hypernym3 in self.get_hypernyms(hypernym2):
          synsets.add(hypernym3)

    synonyms = set()
    synonyms.add(word)
    # print(synonyms)
    for synset in synsets:
      synonyms.update(self.get_synset_word(synset))

    return synonyms



class SynonymizerParser():
  def __init__(self):
    self.cache = {}
    self.sentence_preprocessor = SentencesPreprocessor()

  def get_synonyms_by_site(self, word):
    time.sleep(5)
    url = f'https://sinonim.org/s/{word}#list-s'
    headers = {
      'authority': 'sinonim.org',
      'method': 'GET',
      'path': '/s/%D0%BA%D0%BE%D1%82',
      'scheme': 'https',
      'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
      'accept-encoding': 'gzip, deflate, br',
      'accept-language': 'ru,en;q=0.9,nl;q=0.8',
      'cache-control': 'max-age=0',
      'cookie': '_ym_uid=1666721827903163901; _ym_d=1666721827; _ga=GA1.2.493696382.1677445558; _gid=GA1.2.1430881398.1678980046; _ym_isad=1; num_hits=12',
      'dnt': '1',
      'referer': 'https://sinonim.org/',
      'sec-ch-ua': '"Not?A_Brand";v="8", "Chromium";v="108", "Yandex";v="23"',
      'sec-ch-ua-mobile': '?0',
      'sec-ch-ua-platform': '"Windows"',
      'sec-fetch-dest': 'document',
      'sec-fetch-mode': 'navigate',
      'sec-fetch-site': 'same-origin',
      'sec-fetch-user': '?1',
      'upgrade-insecure-requests': '1',
      'user-agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 YaBrowser/23.1.3.949 Yowser/2.5 Safari/537.36'
    }

    page = requests.post(url, headers=headers)
    soup = BeautifulSoup(page.text, "html.parser")

    synonyms = soup.findAll('a', href=True, id=True)
    synonyms = set(map(lambda syn: syn.text.strip(), synonyms))
    synonyms = set(filter(lambda syn: syn.count(' ') == 0, synonyms))
    synonyms.add(word)

    prep_synonyms = set()
    for syn in synonyms:
      prep_syn = self.sentence_preprocessor.normalize_sentence(syn)
      if prep_syn:
        prep_synonyms.add(prep_syn[0])

    return synonyms

  def get_synonyms(self, word):
    words = SentencesPreprocessor().normalize_sentence(word)
    if len(words) == 0:
      return set()

    word = words[0]
    return self.cache.get(word) if self.cache.get(word, None) else self.cache.setdefault(word, self.get_synonyms_by_site(word))

TRANSLATE_POS = {
    'ADJ': 'Прилагательное',
    'ADV': 'Наречие',
    'INTJ': 'Междометие',
    'NOUN': 'Существительное',
    'PROPN': 'Имя собственное',
    'VERB': 'Глагол',

    'ADP': 'Предикатив',
    'AUX': 'Вспомогательный глагол',
    'CCONJ': 'Сочинительный союз',
    'DET': 'Определитель',
    'NUM': 'Числительное',
    'PART': 'Частица',
    'PRON': 'Местоимение',
    'SCONJ': 'Подчиняющий союз',

    'PUNCT': 'Пунктуация',
    'SYM': 'Символ',
    'X': 'Другое'
}

def get_pos_distr(text) -> Dict:
  count = 0
  pos_dict = {
    'ADJ': 0,
    'ADV': 0,
    'INTJ': 0,
    'NOUN': 0,
    'PROPN': 0,
    'VERB': 0,

    'ADP': 0,
    'AUX': 0,
    'CCONJ': 0,
    'DET': 0,
    'NUM': 0,
    'PART': 0,
    'PRON': 0,
    'SCONJ': 0,

    'PUNCT': 0,
    'SYM': 0,
    'X': 0
}
  for token, pos in get_pos(text):
    # print(token, pos)
    if pos == 'PUNCT':
      continue

    pos_dict[pos] = pos_dict.setdefault(pos, 0) + 1

  return pos_dict

def get_pos_count(text, pos) -> int:
  return get_pos_distr(text).get(pos.upper(), 0)

def get_pos_count_percent(text, pos) -> float:
  count = len(tokenize_wrapper(text))
  return round(get_pos_distr(text).get(pos.upper(), 0) / count, 2)


"""## Синтаксический анализ"""

def build_tokens_list(a_text: str) -> Dict[str, natasha.doc.DocToken]:
    """
    Получение списка токенов из текста. Токены объединены в набор несвязных синтаксических деревьев,
    по одному на каждое предложение. Каждый токен хранит ссылку на своего предка в дереве.

    :param a_text: Входной текст.
    :return: Словарь токенов типа natasha.doc.DocToken. Ключами играют роль номера токенов
    DocToken#id, перевести которые в число невозможно. Ключи отсортированы по встрече
    соответствующих слов в тексте.
    """
    doc = Doc(a_text)
    doc.segment(g_segmenter)
    doc.tag_morph(g_morph_tagger)
    doc.parse_syntax(g_syntax_parser)

    # Составление словаря:
    map: Dict[str, natasha.doc.DocToken] = dict()
    for token in doc.tokens:
        map[token.id] = token

    return map


def draw_syntax_tree(a_tokens: Dict[str, natasha.doc.DocToken], a_drawer: Any, **kwargs) -> None:
    """
    Функция для отрисовки дерева. Принимает последним аргументом функцию, которая занимается
    отрисовкой.

    :param a_text: Список токенов типа natasha.doc.DocToken.
    :param a_drawer: Функция рисования.
    :param kwargs: Именованные аргументы для a_drawer.
    """
    new_id_to_old_id = {}
    counter: int = 0
    for token in a_tokens.values():
        if token.id not in new_id_to_old_id.keys():
            new_id_to_old_id[token.id] = counter
            counter += 1

    words, deps = [], []
    for token in a_tokens.values():
        words.append(token.text)
        source = new_id_to_old_id[token.head_id] if token.head_id in new_id_to_old_id.keys() else -1
        target = new_id_to_old_id[token.id]
        if source > 0 and source != target:
            deps.append([source, target, token.rel])

    a_drawer(words, deps, **kwargs)


def draw_ascii_syntax_tree(a_tokens: Dict[str, natasha.doc.DocToken], **kwargs) -> None:
    """
    Отрисовка синтаксического дерева в виде консольного вывода.

    См. draw_syntax_tree() выше.

    :param a_text: Список токенов типа natasha.doc.DocToken.
    :param kwargs: Именованные аргументы для a_drawer.
    """
    draw_syntax_tree(a_tokens, ipymarkup.show_dep_ascii_markup, **kwargs)


def draw_html_syntax_tree(a_tokens: Dict[str, natasha.doc.DocToken], **kwargs) -> None:
    """
    Отрисовка синтаксического дерева в виде html-картинки.

    :param a_text: Список токенов типа natasha.doc.DocToken.
    :param kwargs: Именованные аргументы для a_drawer.
    """
    draw_syntax_tree(a_tokens, ipymarkup.show_dep_markup, **kwargs)

"""## Поиск именных групп по синтаксическому дереву"""

class Referent:
    """
    Структура для референта. Содержит список именных групп, которые ссылаются на неё.
    """

    def __init__(self, a_first: NounPhrase):
        self.m_noun_phrases: List[NounPhrase] = [a_first]


    def __str__(self) -> str:
        return self.m_noun_phrases[0].head_text()


class NounPhrase:
    """
    Сткрутура для хранения информации по именной группе в необработанном виде, как есть.
    """

    def __init__(
        self,
        a_head: natasha.doc.DocToken,
        a_dom_noun_phrase: Optional[NounPhrase] = None,
        a_dom_verb_phrase: Optional[natasha.doc.DocToken] = None):

        self.m_head: natasha.doc.DocToken = a_head

        # Ближайшая доминирующая именная группа:
        self.m_dom_noun_phrase: Optional[NounPhrase] = a_dom_noun_phrase

        # Ближайшая доминирующая глагольная группа:
        self.m_dom_verb_phrase: Optional[natasha.doc.DocToken] = a_dom_verb_phrase

        # Крайне левый токен именной группы:
        self.m_far_left_word: natasha.doc.DocToken = a_head

        # Крайне правый токен именной группы:
        self.m_far_right_word: natasha.doc.DocToken = a_head

        # Ссылка на именную группу-перечисление, к которой, если, принадлежит данная именная группа:
        self.m_conjunct: Optional[NounPhrase] = None

        # Ссылка на референт:
        self.m_referent: Optional[Referent] = Referent(self)


    def __str__(self) -> str:
        return f'{self.m_head.id} {self.m_head.text}'
        # return f'{self.m_head.id} {self.m_head.text} ' \
        #        f'({self.m_far_left_word.id}:{self.m_far_right_word.id}) ' \
        #        f'{self.m_dom_noun_phrase.text} {self.m_dom_verb_phrase.text}'


    def head_text(self) -> str:
        """
        :return: Запись слова-вершины в виде строки.
        """
        return self.m_head.text


    def head_pos(self) -> str:
        """
        :return: Часть речи вершины.
        """
        return self.m_head.pos


    def head_feats(self) -> List[Dict[str, str]]:
        """
        :return: Грамматические признаки вершины.
        """
        return self.m_head.feats


    def head_rel(self) -> List[Dict[str, str]]:
        """
        :return: Синтаксическая роль вершины.
        """
        return self.m_head.rel


    def sentence_id(self) -> int:
        """
        :return: Номер предложения.
        """
        return int(self.m_head.id.split('_')[0])


    def head_id_in_sentence(self) -> int:
        """
        :return: Позиция вершины в предложении.
        """
        return int(self.m_head.id.split('_')[1])


def is_a_head_of_noun_phrase(a_token: natasha.doc.DocToken) -> bool:
    """
    Проверка, является однозначно ли токен a_token вершиной некой именной группы. Существуют именные
    группы, которые не будут проходить эту проверу: перечисления.

    :param a_token: Входной текст.
    :return: True или является, и False иначе.
    """
    return a_token.rel in ['nsubj', 'nsubj:pass', 'obl', 'iobj', 'obj', 'xcomp', 'appos'] and \
           a_token.pos not in ['PUNCT', 'X', 'ADP', 'ADJ', 'ADV' 'SCONJ', 'DET', 'VERB', 'NUM']


def is_a_head_of_verb_phrase(a_token: natasha.doc.DocToken) -> bool:
    """
    Проверка, является однозначно ли токен a_token вершиной некой глагольной группы. Существуют
    глагольные группы, которые не будут проходить эту проверу: перечисления.

    :param a_token: Входной текст.
    :return: True или является, и False иначе.
    """
    return a_token.rel == 'root' and a_token.pos is 'VERB'


def is_a_conjunct(a_token: natasha.doc.DocToken) -> bool:
    """
    Проверка, является ли токен a_token элементом последовательности перечислений, как правило
    разделённых запятой или союзами 'и' или 'или'.

    :param a_token: Входной текст.
    :return: True или является, и False иначе.
    """
    return a_token.rel == 'conj'


def get_all_noun_phrases(a_tokens: Dict[str, natasha.doc.DocToken]) -> Dict[str, NounPhrase]:
    """
    Поиск всех именных групп в синтаксическом дереве a_tokens при помощи системы правил.

    :param a_tokens: Токены входного текста, отсортированные по порядку встречи соответствующих
    слов.
    :return: Словарь объектов типа NounPhrase -- найденных именных групп, отсортированных по порядку
    встречи их вершин в тексте. Ключами выступают значения DocToken#id вершин соответствующих
    именных групп.
    """

    # Список найденных именных групп:
    res: Dict[str, NounPhrase] = dict()

    # Рекурсивная функция, преследующая цель обнаружить ближайшие доминирующие именную группу и
    # глагольную группу, а так же обновить для первой значения m_far_left_word и m_far_right_word.
    def rec_find_dom(a_token: natasha.doc.DocToken) -> \
        Tuple[Optional[NounPhrase], Optional[natasha.doc.DocToken]]:
        """

        :param a_token: Входной токен.
        :return: Пара из ближайших доминирующих именной и глагольной групп.
        """

        it: Optional[natasha.doc.DocToken] = a_tokens.get(a_token.id)
        while True:
            # Случай отсутствия очередного предка:
            if it is None:
                return None, None
            # Случай, когда очередной предок оказывается вершиной именной группы:
            elif is_a_head_of_noun_phrase(it):

                # Создать объект именной группы если необходимо:
                if it.id in res.keys():
                    np = res[it.id]
                else:
                    try:
                        np = NounPhrase(it, *rec_find_dom(a_tokens.get(it.head_id)))
                        res[it.id] = np
                    except RecursionError:
                        print('rec!')
                        return None, it

                # Обновить m_far_left_word для np:
                if a_token.start < np.m_far_left_word.start:
                    np.m_far_left_word = a_token

                # Обновить m_far_right_word для np:
                if np.m_far_right_word.start < a_token.start:
                    np.m_far_right_word = a_token

                # Обновить m_conjunct для np:
                if is_a_conjunct(a_token):

                    # Создать объект именной группы для a_token:
                    old_np = NounPhrase(a_token)
                    res[a_token.id] = old_np

                    # Создать объект именной группы-перечисления если необходимо:
                    if np.m_conjunct is None:
                        conjunct_np = NounPhrase(it, np.m_dom_noun_phrase, np.m_dom_verb_phrase)
                        # Установить для np доминирующей именной группой conjunct_np:
                        np.m_dom_noun_phrase = conjunct_np
                        res[it.id + '_conjunct'] = conjunct_np
                    else:
                        conjunct_np = np.m_conjunct

                    # Установить и для old_np то же самое что и для np:
                    old_np.m_dom_noun_phrase = conjunct_np
                    old_np.m_dom_verb_phrase = np.m_dom_verb_phrase

                    res[a_token.id].m_conjunct = conjunct_np

                return np, np.m_dom_verb_phrase
            # Случай, когда очередной предок оказывается вершиной глагольной группы:
            elif is_a_head_of_verb_phrase(it):
                return None, it

            # Установить итератор на предка текущего токена:
            it = a_tokens.get(it.head_id)

    # Переберём все токены и проверим, являются ли они вершиной именной группы:
    for token in a_tokens.values():
        rec_find_dom(token)

    return res


def draw_noun_phrases_in_text(
    a_text: str, a_noun_phrases: Dict[str, NounPhrase], a_drawer: Any, **kwargs) -> None:
    """
    Нарисовать текст с обозначенными именными группами. Принимает последним аргументом функцию,
    которая занимается отрисовкой.

    :param a_text: Исходный текст.
    :param a_noun_phrases: Словарь объектов типа NounPhrase, отсортированных по порядку встречи их
    вершин в тексте. Ключами выступают значения DocToken#id вершин соответствующих именных групп.
    :param a_drawer: Функция рисования.
    :param kwargs: Именованные аргументы для a_drawer.
    """

    # Составить список отрезков:
    lines: List[Tuple[int, int, str]] = []
    for np in a_noun_phrases.values():
        if np.m_referent is None:
            lines.append((np.m_far_left_word.start, np.m_far_right_word.stop,
                          np.head_text()))
        else:
            lines.append((np.m_far_left_word.start, np.m_far_right_word.stop,
                          np.m_referent.m_noun_phrases[-1].head_text()))

    a_drawer(a_text, lines, **kwargs)


def draw_ascii_noun_phrases_in_text(
    a_text: str, a_noun_phrases: Dict[str, NounPhrase], **kwargs) -> None:
    """
    Нарисовать текст с обозначенными именными группами в виде консольного вывода.

    :param a_text: Исходный текст.
    :param a_noun_phrases: Словарь объектов типа NounPhrase, отсортированных по порядку встречи их
    вершин в тексте. Ключами выступают значения DocToken#id вершин соответствующих именных групп.
    :param kwargs: Именованные аргументы для a_drawer.
    """
    draw_noun_phrases_in_text(a_text, a_noun_phrases, ipymarkup.show_span_ascii_markup, **kwargs)


def draw_html_noun_phrases_in_text(
    a_text: str, a_noun_phrases: Dict[str, NounPhrase], **kwargs) -> None:
    """
    Нарисовать текст с обозначенными именными группами в виде html-картинки.

    :param a_text: Исходный текст.
    :param a_noun_phrases: Словарь объектов типа NounPhrase, отсортированных по порядку встречи их
    вершин в тексте. Ключами выступают значения DocToken#id вершин соответствующих именных групп.
    :param kwargs: Именованные аргументы для a_drawer.
    """
    draw_noun_phrases_in_text(a_text, a_noun_phrases, ipymarkup.show_span_line_markup, **kwargs)



class SyntaxTree:
  def __init__(self, text):
    self.doc = Doc(text)
    self.doc.segment(segmenter)
    self.doc.parse_syntax(syntax_parser)

    self.count_sents = len(self.doc.sents)
    print(self.doc.text)

  def get_roots_by_sent(self, n_sent):
    roots = []
    for token in self.doc.sents[n_sent].syntax.tokens:
      if token.rel == 'root':
        roots.append((token.id, token.text))

    return roots

  def print_tree(self, n_sent: int):
    self.doc.sents[n_sent].syntax.print()

  def get_num(self, id: str):
    return int(id[id.index('_')+1:])

  def get_ids_by_text(self, text: str, n_sent: int):
    ids = []
    for token in self.doc.sents[n_sent].syntax.tokens:
      if token.text == text:
        ids.append(self.get_num(token.id))
    return ids

  def token_parents(self, token_text: str, n_sent: int):
    # TODO переделать на id
    parents = []
    for token in self.doc.sents[n_sent].syntax.tokens:
      if token.text == token_text and token.rel != 'root':
        parents.append(self.doc.sents[n_sent].syntax.tokens[self.get_num(token.head_id)])
    return parents

  def token_children(self, token_id: str, n_sent: int):
    children = []
    for child_token in self.doc.sents[n_sent].syntax.tokens:
      if child_token.head_id == token_id and child_token.rel != 'punct':
        children.append((child_token.id, child_token.text))
    return children

  def get_rels(self, depth: int):
    def rec_get_rels(rels: List, word_id: str, n_sent: int, prev_words: List):
      children = self.token_children(word_id, n_sent)
      if children:
        for child_id, child_text in children:
          if len(prev_words) == depth-1:
            rels.append(prev_words + [child_text.lower()])
            rec_get_rels(rels, child_id, n_sent, prev_words[1:] + [child_text.lower()])
          else:
            rec_get_rels(rels, child_id, n_sent, prev_words + [child_text.lower()] )

    rels = []

    if depth == 1:
      for token in self.doc.tokens:
        if token.rel != 'punct':
          rels.append([token.text.lower()])
    else:
      for n_sent in range(self.count_sents):
        for root_id, root_text in self.get_roots_by_sent(n_sent):
          rec_get_rels(rels, root_id, n_sent, [root_text.lower()])

    return rels

  def test(self):
    for n_sent in range(self.count_sents):
      print(tree.doc.sents[n_sent])



def calc_semantic_completeness(text, semantic_components):
  # semantic_components = ['птица', 'мама', 'птенцы', 'кот', 'схватить', 'кормит', 'отвернулась', 'собака', 'схватила', 'спасла', 'прогнала'] # 0.73
  # semantic_components = ['девочка', 'мама', 'мальчик', 'чашки', 'стол', 'осколки', 'ругать', 'девочку', 'прячется', 'блюдце']
  # «Гнездо» 1 птица-мама;  2 птенцы;  3 котик;  4 хочет схватить птенца;  5 кормит птенцов;  6 отвернулась;  7 собака;  8 схватила кота;  9 спасла птенцов;  10 прогнала кота)
  def compare_lists(l1, l2):
    if len(l1) != len(l2):
      raise Exception(f'Длина списков не совпадает: {len(l1)} != {len(l2)}')

    for i in range(len(l1)):
      if l1[i] == l2[i]:
        continue
      w1, w2 = (l1[i], l2[i]) if len(l1[i]) < len(l2[i]) else (l2[i], l1[i])

      if w2 not in list(map(lambda prefix: prefix + w1, DEFAULT_PREFIXES)):
        return False

    return True

  def get_comp_synonyms(comps):
    if type(comps) is not list:
      comps = [comps]
    synonymizer = synonymizer_parser

    comp_synonyms = []
    for comp in comps:
      for comp in product(*[token.split('/') for token in comp.split(' ')]):
        comp_synonyms_list = [synonymizer.get_synonyms(comp_word) for comp_word in comp]

        # print(comp_synonyms_list)
        for comp_synonym in product(*comp_synonyms_list):
          comp_synonyms.append(list(comp_synonym))

    return comp_synonyms

  sent_preprocess = SentencesPreprocessor()
  norm_text = sent_preprocess.normalize_sentence(text)
  # text = text.replace('.', ',')

  count = 0

  tree = SyntaxTree(text)
  rels_dict = {}
  for i in range(1, 4):
    rels_dict[i] = []
    for rel in tree.get_rels(i):
      prep_rel = sent_preprocess.normalize_iter(rel)
      if len(prep_rel) == i:
        rels_dict[i].append(prep_rel)

  print(rels_dict[2])
  for comp in semantic_components:
    find_flag = False
    for prep_comp_synonym in get_comp_synonyms(comp):
      # print(f'prep_comp_synonym = {prep_comp_synonym}')
      for prep_rel in rels_dict.get(len(prep_comp_synonym), []):
        # prep_rel = sent_preprocess.normalize_iter(rel)
        # print(f'--{prep_rel} ||| {prep_comp_synonym}')
        # stem_prep_rel = list(map(stemmer.stem, prep_rel))
        # stem_prep_comp_synonym = list(map(stemmer.stem, prep_comp_synonym))
        # print(f'--{stem_prep_rel} ||| {stem_prep_comp_synonym}')

        if len(prep_comp_synonym) and len(prep_rel) and compare_lists(prep_rel, prep_comp_synonym):
          print(f'--{prep_rel} ||| {prep_comp_synonym}')
          count += 1
          find_flag = True
          break
      if find_flag:
        break

  return round(count/len(semantic_components), 2)


def count_compound_sentences(tell):
  count_sentences = len(sentenize_wrapper(tell))
  count_compound_sentences = len([sent for sent in sentenize_wrapper(tell) if sent.count(',') > 0])
  return count_compound_sentences

def count_compound_sentences_percent(tell):
  count_sentences = len(sentenize_wrapper(tell))
  count_compound_sentences = len([sent for sent in sentenize_wrapper(tell) if sent.count(',') > 0])
  return round(count_compound_sentences/count_sentences, 2)


"""## 8. Средняя длина фразы (DONE)

"""

def mean_sentences_len(tell):
  sentences_len = list(map(lambda x: len(tokenize_wrapper(x)), sentenize_wrapper(tell)))
  return np.mean(sentences_len)


def ttr(tell):
  return round(len(get_word_uniq(tell))/len(preprocess(tell)), 2)



def create_attrs(tell, semantic_components):
  df = pd.concat([tell, tell.apply(lambda s: len(tokenize_wrapper(s)))], axis=1)
  df = pd.concat([df, tell.apply(lambda s: len(sentenize_wrapper(s)))], axis=1)
  df = pd.concat([df, tell.apply(lambda s: len(get_word_uniq(s)))], axis=1)
  df = pd.concat([df, tell.apply(count_compound_sentences_percent)], axis=1)
  df = pd.concat([df, tell.apply(mean_sentences_len)], axis=1)
  df = pd.concat([df, tell.apply(ttr)], axis=1)
  df = pd.concat([df, tell.apply(lambda s: get_pos_count_percent(s, "ADJ"))], axis=1)
  df = pd.concat([df, tell.apply(lambda s: get_pos_count_percent(s, "ADV"))], axis=1)
  df = pd.concat([df, tell.apply(lambda s: get_pos_count_percent(s, "PRON"))], axis=1)
  df = pd.concat([df, tell.apply(lambda s: get_pos_count_percent(s, "NOUN"))], axis=1)
  df = pd.concat([df, tell.apply(lambda s: get_pos_count_percent(s, "VERB"))], axis=1)
  df = pd.concat([df, tell.apply(lambda s: calc_semantic_completeness(s, semantic_components[0]))], axis=1)
  df = pd.concat([df, tell.apply(lambda s: calc_semantic_completeness(s, semantic_components[1]))], axis=1)
  df = pd.concat([df, tell.apply(lambda s: calc_semantic_completeness(s, semantic_components[2]))], axis=1)

  df.columns = [tell.name] + ['Кол-во слов', 'Кол-во предл', 'Кол-во уник. слов', 'Кол-во слож предл', 'Средняя длина предл',
                'TTR', 'Проц прил', 'Проц наречий', 'Проц мест', 'Проц сущ', 'Проц глаг', 'Семант полнота',
                'Сущ сем комп', 'Действия сем комп']

  return df

def get_df(df, postfix, semantic_components):
  df_cor = create_attrs(df[f'tell{postfix}'], semantic_components).iloc[:, 1:]
  df_cor['time'] = df[f'time{postfix}']
  df_cor['is_strong'] = df['is_strong']
  return df_cor

def get_clusters(df_cor):
  scaler = preprocessing.StandardScaler().fit(df_cor)
  df_scaled = scaler.transform(df_cor)

  clustering = KMeans(n_clusters=2, random_state=0).fit_predict(df_scaled)
  # clustering = SpectralClustering(n_clusters=2, n_neighbors=3, n_init=2, affinity='nearest_neighbors').fit_predict(df_scaled)

  return clustering

def get_heatmap(df_cor):
  plt.figure(figsize=(10,5))
  sns.heatmap(df_cor.corr(), annot = True, vmin=-1, vmax=1, center= 0)

DEFAULT_PREFIXES = ('а','агит','ан','англо','анти','атто','без','бес','брам','в','вз','вне','военно','воз','вос','вы','гекса','гексаконта','гекта','гекто','гепта','гептаконта','гига',
'гипер','гор','гос','де','дез','дека','деци','дикта','до','додека','за','зепто','зетта','из','изо','ин','интервики','интра','инфра','йокто','йотта','квадра','квази','кила','кило',
'ко','кое','контр','лейб','мега','меж','микро','милли','мини','мириа','моно','на','над','наи','нано','не','недо','ни','низ','нис','нона','о','около','окта','октаконта','от','па','пара',
'пентаконта','пере','пета','пико','по','под','после','пост','пра','пре','пред','при','про','прото','раз','ре','роз','рос','с','санти','сверх','со','су','суб','супер','супра','сюр','тера',
'тетра','тетраконта','транс','тре','три','триаконта','тридека','трикта','у','ультра','ундека','фемто','черес','эйкоза','экзо','экс','экса','экстра','эннеаконта')

from flask import Flask, request

X_train = pd.DataFrame()
feature_names = ''
X_train_scaled = pd.DataFrame()

app = Flask(__name__)

@app.route('/api/nlp', methods=['POST'])
def index():
    global X_train
    global feature_names
    global X_train_scaled

    json = request.get_json()

    df1 = pd.DataFrame({"time1": json["time1"], "is_strong": ["1"], "tell1": json["tell1"]})

    df2 = pd.DataFrame({"time2": json["time2"], "is_strong": ["1"], "tell2": json["tell2"]})

    df3 = pd.DataFrame({"time3": json["time3"], "is_strong": ["1"], "tell3": json["tell3"]})

    Semantic_components = namedtuple('Semantic_components', ['full' , 'entities', 'actions'])

    semantic_components_full = ['мама/мать спросила/сказала', 'дети/мальчик и девочка вернулись/пришли школы',
                                'мальчик/мама/девочка хотел/знал/решил/заказал', 'оставила/забыла сумку/кошелек/деньги']
    semantic_components_entities = ['мама/мать', 'дети' 'девочка', 'мальчик', 'еда', 'чизбургер', 'сумка/кошелек',
                                    'Макдональдс/ресторан', 'дома', 'чизбургер/сэндвич', 'коктейль', 'мороженое/рожок',
                                    'салат', 'кола', 'хэппи мил', 'школа', 'продавец/кассир']
    semantic_components_actions = ['вернулись', 'спросила/сказала', 'кричали' 'сели', 'поехали/пошли',
                                   'не знала/решила', 'заказать', 'хотел/знали', 'забыла/оставила']
    semantic_components1 = Semantic_components(semantic_components_full, semantic_components_entities,
                                               semantic_components_actions)

    semantic_components_full = ['мальчик проснулся/встал', 'посмотрел время/часы', 'разлил молоко',
                                'опоздал школу/автобус', 'порвались/сломались шнурки/веревки', 'автобус ушел/уехал']
    semantic_components_entities = ['мальчик', 'хлопья', 'молоко', 'шнурки/веревки', 'ботинки', 'автобус', 'школа',
                                    'учитель', 'время/часы']
    semantic_components_actions = ['проснулся', 'посмотрел', 'разлил', 'порвал/сломал', 'опоздал', 'побежал/пошел']
    semantic_components2 = Semantic_components(semantic_components_full, semantic_components_entities,
                                               semantic_components_actions)

    semantic_components_full = ['мальчик/брат напугал/испугался', 'девочка/сестра посмотреть/подружиться/увидеть',
                                'нло/инопланеняне/пришельцы упала/приземлилась/прилетела']
    semantic_components_entities = ['мальчик/брат', 'девочка/сестра', 'нло/инопланеняне/пришельцы',
                                    'космический корабль', 'мама', 'папа', 'собака', 'люди', 'дети/ребенок', 'родители']
    semantic_components_actions = ['гуляли', 'увидели', 'упала/приземлилась/прилетела', 'посмотреть/подружиться',
                                   'испугался/боялся', 'спрятался', 'вернулись', 'не поверили']
    semantic_components3 = Semantic_components(semantic_components_full, semantic_components_entities,
                                               semantic_components_actions)


    df_new = pd.concat([get_df(df1, "1", semantic_components1),
                        get_df(df2, "2", semantic_components2),
                        get_df(df3, "3", semantic_components3)], axis=1)

    new_columns = ['Кол-во слов', 'Кол-во предл', 'Кол-во уник. слов', 'Кол-во слож предл',
                   'Средняя длина предл', 'TTR', 'Проц прил', 'Проц наречий', 'Проц мест',
                   'Проц сущ', 'Проц глаг', 'Семант полнота', 'Сущ сем комп',
                   'Действия сем комп', 'time', 'is_strong', 'Кол-во слов.1',
                   'Кол-во предл.1', 'Кол-во уник. слов.1', 'Кол-во слож предл.1',
                   'Средняя длина предл.1', 'TTR.1', 'Проц прил.1', 'Проц наречий.1',
                   'Проц мест.1', 'Проц сущ.1', 'Проц глаг.1', 'Семант полнота.1',
                   'Сущ сем комп.1', 'Действия сем комп.1', 'time.1', 'is_strong.1',
                   'Кол-во слов.2', 'Кол-во предл.2', 'Кол-во уник. слов.2',
                   'Кол-во слож предл.2', 'Средняя длина предл.2', 'TTR.2', 'Проц прил.2',
                   'Проц наречий.2', 'Проц мест.2', 'Проц сущ.2', 'Проц глаг.2',
                   'Семант полнота.2', 'Сущ сем комп.2', 'Действия сем комп.2', 'time.2',
                   'is_strong.2']

    df_new.columns = new_columns

    df_new = df_new.drop(['is_strong', 'is_strong.1', 'is_strong.2'], axis=1, errors='ignore')

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_scaled,
        feature_names=feature_names,
        class_names=['0', '1'],
        mode='classification'
    )

    all_features = X_train.columns.tolist()

    for feature in df_new.columns:
        if feature in df.columns:
            df_new[feature] = df[feature].values[0]

    new_scaler = scaler.transform(df_new)

    instance = new_scaler[0].reshape(1, -1)
    prediction = model.predict_proba(instance)

    return str(prediction[0][1])

if __name__ == "__main__":
    segmenter = Segmenter()
    morph_vocab = MorphVocab()
    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)
    syntax_parser = NewsSyntaxParser(emb)
    synonymizer_parser = SynonymizerParser()

    sent_preprocess = SentencesPreprocessor()
    preprocess = SentencesPreprocessor().normalize_sentence

    nltk.download("stopwords")

    data = pd.read_csv(
        'data_ch_new.csv',
        header=1)
    data = data.drop(['is_strong.1', 'is_strong.2'], axis=1, errors='ignore')
    df = pd.DataFrame(data)
    X_train = df.drop('is_strong', axis=1)

    feature_names = X_train.columns.tolist()
    X_train_df = pd.DataFrame(X_train, columns=feature_names)

    scaler = preprocessing.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)

    model = joblib.load('xgb_clf2.joblib')

    app.run(debug=False, port=5000)


