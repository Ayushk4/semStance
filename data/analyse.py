import json

fo = open("data/pre_processed.json", "r")
dataset = json.load(fo)
fo.close()

def text_length_statishtic(all_data):
  mini = 10000000
  mini_id = ""
  maxi = 0
  maxi_id = ""
  avg = 0
  
  for data in all_data:
    this_len = len(data["text"])
    if this_len <= 3:
      print(data.values(), "\n")
    if this_len > 0 and mini > this_len:
      mini_id = data["tweet_id"]
      mini = this_len
    if maxi < this_len:
      maxi_id = data["tweet_id"]
      maxi = this_len
    avg += this_len

  return mini, mini_id, maxi, maxi_id, (avg/len(all_data))

if __name__ == "__main__":
  print(text_length_statishtic(dataset))



