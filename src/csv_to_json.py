import sys
sys.path.append('../')

import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

data_path     = Path('../data/')
csv_file_path = Path('../data/otquad-v1.0 - Sheet1.csv')

df = pd.read_csv(csv_file_path)

start_index = [df['context'].values[i].lower().find(df['answer'].values[i].lower()) for i in range(len(df))]
df['start_index'] = start_index

for i in range(len(df)):
    print(i, start_index[i])
    print('Original : ',df['answer'].values[i].lower())
    print('Extracted: ', df['context'].values[i].lower()[start_index[i]:start_index[i]+len(df['answer'].values[i])])
    assert df['answer'].values[i].lower() == df['context'].values[i].lower()[start_index[i]:start_index[i]+len(df['answer'].values[i])]
    
    
print(f'Unique Contexts: {len(np.unique(df.context.values))}\n Ratio of QA Pair-to-Contexts: {len(df)/len(np.unique(df.context.values))}')


squad_entries=[]
for i in tqdm(range(len(df))):
    id_num      = i
    context     = df['context'].values[i]
    question    = df['question'].values[i]
    answer_text = df['answer'].values[i]
    start       = df['start_index'].values[i]
    
    if start!=-1:
        answer={"text":answer_text,"answer_start":int(start)}
    else:
        continue
    new_entry={
               "qas": [
                 {
                  "id": int(id_num),
                  "question": question,
                  "answers": [answer]
                 }
                 ],
               "context": context
               }
    squad_entries.append(new_entry)
    
    
def entry_to_json(squad_entries,title=None,version=None):
    squad_json={
              "data": [
                {
                  "paragraphs":squad_entries,
                   "title":title
                }],
               "version":version
           }
    return squad_json


otquad_json = entry_to_json(squad_entries,title=None,version=1.0)


json_file_name = "otquad-v1.0.json"

with open(data_path/json_file_name, "w") as writer:
    writer.write(json.dumps(otquad_json, indent=4) + "\n")
