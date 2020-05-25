# Input: data/wtwt_ids.json and data/scrapped_full/*
# Output: data/wtwt_obtained.json and data/pre_processed.json

# Read all existing files into a dict mapping id to full_text
#   Then preprocess, tokenize the dict text and save pre-processed dicts
#   Remove the tweet ids not present in wtwt_ids.json into wtwt_obtained.json


