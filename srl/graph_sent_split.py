# Before SRL:
    # Sentence Split about ":" is safe
    # Sentence Split about ";" is safe
    # Sentence Split about "?" is safe
    # Sentence Split about "!" is safe
    # Sentence split about "." not after [No, St, ... ] check for token == "." rather than "." in token
    # Handling <user> and <hashtags>

# Also prepare input text for LSTM example: re.sub(" sasasasasas", "'s", period_splits)

import json
import re

fo = open("non_breaking_prefixes.txt", "r")
prefixes_nb_period = list(map(lambda x: x.strip(), fo.readlines()))
fo.close()

PERIOD_SPACE_SPLIT_KEY = "\t\nss\n\t"
COMPANY_NAMES = ["Cvs", "Avetna", "Cvgna", "Exprsx", "Antema", "Huumana"]

def sentence_wise_srl(tokens, tid):
    # re.sub from " sasasasasas" => "'s"
    # split sentence about : ? ; ! | and non-breaking "."
    text = " ".join(tokens)
    text = re.sub(" sasasasasas", "'s", text)
    text = re.sub("sasasasasas", "'s", text)

    sentences = re.split(r"([:;?!\|])", text)
    final_sentences = []
    sent_idx = 0
    while sent_idx < len(sentences):
        sent = sentences[sent_idx].split()
        sent_idx += 1

        if sent_idx < len(sentences) - 1:
            if sentences[sent_idx] in [":", ";", "|"]:
                sent.append(" . ")
            else:
                sent.append(sentences[sent_idx])
            sent_idx += 1

        idxs = [i for i, e in enumerate(sent) if e == '.']
        for idx in idxs:
            if idx > 0 and sent[idx-1] in prefixes_nb_period:
                pass
            else:
                sent[idx] = PERIOD_SPACE_SPLIT_KEY

        period_splits = " ".join(sent).split(PERIOD_SPACE_SPLIT_KEY)
        prd_split_idx = 0
        while prd_split_idx < (len(period_splits) -1):
            final_sentences.append(period_splits[prd_split_idx].strip() + " .")
            prd_split_idx += 1
        final_sentences.append(period_splits[prd_split_idx].strip())

    final_sentences = list(filter(lambda x: x.strip() not in ["", ".", "?", "!"], final_sentences))

    # print("==============")
    # print(tid, text)
    # print(sentences)
    # print(final_sentences)
    # print("================", "\n")

    fs_with_tag = []
    for _sent in final_sentences:
        _sent = re.sub("</hashtag>'s", "</hashtag>", _sent)
        s = _sent.split()
        if len(s) < 3 or (len(s) == 3 and s[2].strip() == "."):
            fs_with_tag.append(("no-srl", re.sub(r"\$", "",_sent).split()))
            continue
        if "<hashtag>" in s:
            assert "</hashtag>" in s
        if "</hashtag>" in s:
            assert "<hashtag>" in s
            
        if "<hashtag>" not in s:
            fs_with_tag.append(("srl", re.sub(r"\$", "",_sent).split()))
        else:
            only_hashtag = True
            i = 0
            s_tags_removed = []
            while i < len(s):
                if s[i] in ["Uriah", "<user>", ".", "!", "?", ",", "s"] or s[i] in COMPANY_NAMES:
                    s_tags_removed.append(s[i])
                    i += 1
                elif s[i][0] == "$":
                    s_tags_removed.append(s[i][1:])
                    i += 1
                elif s[i] == "<hashtag>":
                    i += 1
                    while s[i] != "</hashtag>":
                        s_tags_removed.append(s[i])
                        i += 1
                    i += 1
                else:
                    only_hashtag = False
                    s_tags_removed.append(s[i])
                    i += 1

            if only_hashtag:
                fs_with_tag.append(("no-srl", s_tags_removed))
            else:
                fs_with_tag.append(("srl", s_tags_removed))

    return fs_with_tag

fo = open("data/pre_processed.json", "r")
preprocessed_data = json.load(fo)
fo.close()

i = 0
splitted_data = []
for tweet in preprocessed_data:
    copy_ = tweet.copy()
    copy_["sent_split"] = sentence_wise_srl(tweet["text"], tweet["tweet_id"])#, tweet["stance"], tweet["raw_text"])
    splitted_data.append(copy_)
    i += 1

print(i)

fo = open("data/sent_splitted.json", "w+")
json.dump(splitted_data, fo, indent=2)
fo.close()

