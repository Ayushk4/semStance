# Preprocess before SRL
# Ends in via <user> <hashtag>
# Capitalize target mergers
# Remove statements ending with via <user> Hashtags and links or news visa <user>....
# Merger consecutive listing of company names into one.

import json
import os
import sys
import glob

import re
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

EMOJI_PATTERN = re.compile("["
         u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE)

text_processor = TextPreProcessor(
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'url', 'date', 'number'],
    annotate={"hashtag", "allcaps", "elongated", "repeated",
        'emphasis', 'censored'},

    fix_html=True,
    segmenter="twitter",
    corrector="twitter",

    unpack_hashtags=True,
    unpack_contractions=True,
    spell_correct_elong=True,

    tokenizer=SocialTokenizer(lowercase=False).tokenize,
    dicts=[emoticons]
)

REMOVE_TAGS = [
    "<emphasis>", "<kiss>", "<repeated>", "<laugh>", "<allcaps>",
    "</allcaps>", "<angel>", "<elongated>", "<tong>", "<annoyed>",
    "<censored>", "<happy>", "<percent>", "<wink>",
    "<headdesk>", "<surprise>", "<date>", "<time>", "<url>",
    "<sad>", "<email>", "<phone>"
    ]

HASHTAGS_CODE = ["<hashtag>", "</hashtag>"]

ADD_TO_GLOVE = ["<number>", "<money>", "<user>"]

PUNCTS = '''()—-[]{\}<>/@#'%"^*_~ +=`'''

def decontracted(phrase):
  phrase = re.sub(r"’", r"\'", phrase)
  phrase = re.sub(r"won\'t", "will not", phrase)
  # phrase = re.sub(r"won\’t", "will not", phrase)
  phrase = re.sub(r"can\'t", "can not", phrase)
  # phrase = re.sub(r"can\’t", "can not", phrase)
  phrase = re.sub(r",", " , ", phrase)
  phrase = re.sub(r"n\'t", " not", phrase)
  # phrase = re.sub(r"n\’t", " not", phrase)
  phrase = re.sub(r"\'re", " are", phrase)
  # phrase = re.sub(r"\’re", " are", phrase)
  # To be reverted in graph_sent_split
  phrase = re.sub(r"\'s", "sasasasasas", phrase)
  # phrase = re.sub(r"\’s", "sasasasasas", phrase)
  phrase = re.sub("\n\n", " . ", phrase)
  phrase = re.sub(r"\'d", " would", phrase)
  # phrase = re.sub(r"\’d", " would", phrase)
  phrase = re.sub(r"\'ll", " will", phrase)
  # phrase = re.sub(r"\’ll", " will", phrase)
  phrase = re.sub(r"\'t", " not", phrase)
  # phrase = re.sub(r"\’t", " not", phrase)
  # phrase = re.sub(r"\’ve", " have", phrase)
  phrase = re.sub(r"\'ve", " have", phrase)
  # phrase = re.sub(r"\’m", " am", phrase)
  phrase = re.sub(r"\'m", " am", phrase)
  phrase = re.sub(r"1st", " first ", phrase)
  phrase = re.sub(r"2nd", " second ", phrase)
  phrase = re.sub(r"3rd", " third ", phrase)
  phrase = re.sub(r"—", " ", phrase)
  phrase = re.sub(r"-", " ", phrase)
  phrase = re.sub(r"->", " | ", phrase)
  return phrase

COMPANY_NORMALIZE = {
                    # Hashtags:
                      r"#CVS": "Cvs",
                      r"#CVSHealth": "Cvs",
                      r"#Aetna": "Avetna",
                      r"#Aet": "Avetna",
                      r"#cigna": "Cvgna",
                      r"#Ci": "Cvgna",
                      r"#expressscripts": "Exprsx",
                      r"#expressscript": "Exprsx",
                      r"#esrx": "Exprsx",
                      r"#antheminc": "Antema",
                      r"#anthemhealth": "Antema",
                      r"#anthem": "Antema",
                      r"#antm": "Antema",
                      r"#antx": "Antema",
                      r"#humana": "Huumana",
                      r"#hum": "Huumana",

                    # Cashtags: from https://stocks.tradingcharts.com/stocks/symbols/s
                      r"\$CVS": "Cvs",
                      r"\$AET": "Avetna",
                      r"\$CIG": "Cvgna",
                      r"\$CI": "Cvgna",
                      r"\$ESRX": "Exprsx",
                      r"\$ANTX": "Antema",
                      r"\$ANTM": "Antema",
                      r"\$HUM": "Huumana",

                    # User_mentions:
                      r"@CVSHealth": "Cvs",
                      r"@Aetna": "Avetna",
                      r"@Cigna": "Cvgna",
                      r"@ExpressScripts": "Exprsx",
                      r"@ExpressRxHelp": "Exprsx",
                      r"@AnthemInc": "Antema",
                      r"@HumanaHelp": "Huumana",
                      r"@Humana": "Huumana",

                    # Company names or aacronymns
                      r"CVS Health": "Cvs",
                      r"CVSHealth": "Cvs",
                      r"Aetna health": "Avetna",
                      r"Aetnahealth": "Avetna",
                      r"Cigna": "Cvgna",
                      r"Express Scripts": "Exprsx",
                      r"ExpressScripts": "Exprsx",
                      r"Esrx": "Exprsx",
                      r" Esi": "Exprsx",
                      r"Anthem, Inc.": "Antema",
                      r"Anthem, Inc.": "Antema",
                      r"Anthem Inc.": "Antema",
                      r"Anthem, Inc": "Antema",
                      r"Anthem Inc": "Antema",
                      r"AnthemInc": "Antema",
                      r"Anthem Health": "Antema",
                      r"AnthemHealth": "Antema",
                      r"humana": "Huumana",

                  # Capitalize so that ekphrasis doesn't split
                      r"CVS": "Cvs",
                      r"(^|[\/\\\- ])ci($|[\/\\\- ])": " Cvgna ",
                      r"Aetna": "Avetna",
                      r"Aet": "Avetna",
                      r"Anthem": "Antema",
                      r"Antm": "Antema",
                      r" hum ": "Huumana",
                    }

COMPANY_NAMES_Both_case = ["Cvs", "Avetna", "Cvgna", "Exprsx", "Antema", "Huumana",
                            "cvs", "avetna", "cvgna", "exprsx", "antema", "huumana"]

CLEANED_REMAP = {
            "cvs" : "Cvs",
            "avetna": "Avetna",
            "cvgna": "Cvgna",
            "exprsx": "Exprsx",
            "antema": "Antema",
            "huumana": "Huumana",
            "i": "I",
            "<money>": "1",
            "<number>": "0",
            "<user>": "Uriah"
}

def normalize_companies(text):
  for regex in COMPANY_NORMALIZE.keys():
    normalized = COMPANY_NORMALIZE[regex]
    text = re.sub(regex, " " + normalized + " ",
                  text, flags=re.IGNORECASE)
  return text

def pre_process_single(tweet, t_id):
  tweet_toked_text = []
  de_emojified_text = decontracted(tweet)
  de_emojified_text = de_emojified_text.encode('ascii', 'ignore').decode('ascii')
  de_emojified_text = EMOJI_PATTERN.sub(r' ', de_emojified_text)
  de_emojified_text = decontracted(de_emojified_text)
  company_normalize_text = normalize_companies(de_emojified_text)

  tokens = text_processor.pre_process_doc(company_normalize_text)

  flag_via=False
  t = 0
  if len(tokens) > t and tokens[t] == "via":
    t+=1
    flag_via = True

  if len(tokens) > t + 1 and tokens[t] == "news" and tokens[t+1] == "via":
    t+=2
    flag_via = True

  if flag_via == True:
    while t < len(tokens) and (tokens[t] == "<user>" or tokens[t] in REMOVE_TAGS):
      t += 1

  while t < len(tokens):
    flag_via = False
    if len(tokens) > t + 1 and tokens[t] == "via" and tokens[t+1] == "<user>":
      t+=2
      flag_via = True

    if len(tokens) > t + 2 and tokens[t] == "news" and tokens[t+1] == "via" and tokens[t+2] == "<user>":
      t+=3
      flag_via = True

    if flag_via == True:
      while t < len(tokens) and (tokens[t] == "<user>" or tokens[t] in REMOVE_TAGS):
        t += 1

      tweet_toked_text.append("|")
      continue

    token = tokens[t]
    t+=1

    if token in REMOVE_TAGS:
      if token == "<url>": # Divide sentence about URL
        tweet_toked_text.append("|")
    elif token in HASHTAGS_CODE:
      tweet_toked_text.append(token)
    elif token == "<user>":
      tweet_toked_text.append(CLEANED_REMAP["<user>"]) # Proper noun not present in tweets.    
    else:
      not_punct = True
      if token not in ADD_TO_GLOVE:
        for p in PUNCTS:
          if p in token:
            not_punct = False
            break

      if not_punct == True:
        if token.isdigit():
          tweet_toked_text.append("0")
        elif token[0] == "$":
          if token == "$":
            pass
          else:
            tweet_toked_text.append("$" + CLEANED_REMAP.get(token[1:], token[1:]))
            # tweet_toked_text.append(CLEANED_REMAP.get(token[1:], token[1:]))
        else:
          tweet_toked_text.append(CLEANED_REMAP.get(token, token))

  return tweet_toked_text

DOUBLE_NORMED_TOKENS = ["Cvs", "Avetna", "Cvgna", "Exprsx", "Antema", "Huumana",
                  CLEANED_REMAP["<user>"], CLEANED_REMAP["<number>"], CLEANED_REMAP["<money>"]]

def collapse_double(toked_text, tid):
  collapsed_tokens = []
  if len(toked_text) == 0:
    print(tid)
    return []
  i = 0
  collapsed_tokens.append(toked_text[0])
  i += 1
  while i < len(toked_text):
    if toked_text[i] in DOUBLE_NORMED_TOKENS and toked_text[i-1] == toked_text[i]:
      i += 1
      continue
    collapsed_tokens.append(toked_text[i])
    i += 1
    
  return collapsed_tokens

id2text = {}
id2raw_text = {}

for fil in glob.glob("../data/data/scrapped_full/*"):
  fo = open(fil, "r")
  full_tweet = json.load(fo)
  fo.close()

  tweet_id = full_tweet["id_str"]

  txt = collapse_double(pre_process_single(full_tweet["full_text"], full_tweet["id"]), full_tweet["id"])
  if len(txt) > 0 and txt != ["<user>"]:
    id2text[tweet_id] = txt
    id2raw_text[tweet_id] = full_tweet["full_text"]
  else: 
    pass#print(txt, tweet_id)
fo = open("../data/data/wtwt_ids.json", "r")
wtwt = json.load(fo)
fo.close()

all_keys = id2text.keys()
wtwt_obtained = []

for data in wtwt:
  if data["tweet_id"] in all_keys:
    d = data.copy()
    d["text"] = id2text[data["tweet_id"]]
    d["raw_text"] = id2raw_text[data["tweet_id"]]
    wtwt_obtained.append(d)

fo = open("data/pre_processed.json", "w+")
json.dump(wtwt_obtained, fo, indent=2)
fo.close()

