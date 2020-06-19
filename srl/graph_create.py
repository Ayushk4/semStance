import sys
assert len(sys.argv) == 3

import json
from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction.models.srl
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.03.24.tar.gz")

print("Loaded SRL Model")
def single_tweet_graph(sent_split, tid):
    srl_graphed = []
    for sent_type, sentence in sent_split:
        if sent_type == "srl":
            srl_graphed.append(predictor.predict(" ".join(sentence)))
        else:
            srl_graphed.append(sentence)

    return srl_graphed
fo = open("data/sent_splitted.json", "r")
preprocessed_data = json.load(fo)
fo.close()

from_ = max(0, int(sys.argv[1]))
to_ = min(int(sys.argv[2]), len(preprocessed_data))

i = from_
srl_tweet = []
for tweet in preprocessed_data[from_:to_]:
    copy_ = tweet.copy()
    copy_["srl_graphs"] = single_tweet_graph(tweet["sent_split"], tweet["tweet_id"])
    srl_tweet.append(copy_)
    if i % 10 == 0:
        print(i)
    i += 1

print(i)
 
fo = open("data/srl" + str(from_) + "_" + str(to_) + ".json", "w+")
json.dump(srl_tweet, fo, indent=2)
fo.close()
# print(predictor.predict("The keys, which were needed to access the building, were locked in the car."))
