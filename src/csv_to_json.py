import sys
sys.path.append('../')

import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

data_path     = Path('../data/')
csv_file_path = Path('../data/cpgQA-v1.0.csv')

df = pd.read_csv(csv_file_path)
df = df[['context','question','answer','title']]

start_index = [df['context'].values[i].lower().find(df['answer'].values[i].lower()) for i in range(len(df))]
df['start_index'] = start_index

for i in range(len(df)):
    print(i, start_index[i])
    print('Original : ',df['answer'].values[i].lower())
    print('Extracted: ', df['context'].values[i].lower()[start_index[i]:start_index[i]+len(df['answer'].values[i])])
    assert df['answer'].values[i].lower() == df['context'].values[i].lower()[start_index[i]:start_index[i]+len(df['answer'].values[i])]
    
    
print(f'Unique Contexts: {len(np.unique(df.context.values))}\n Ratio of QA Pair-to-Contexts: {len(df)/len(np.unique(df.context.values))}')


new_title = {'Discussion of Recommendations: Risk Mitigation: Recommendation':'Recommendations',
 'Discussion of Recommendations: Risk Mitigation: Discussion: Written Informed Consent and Opioid Treatment Agreements \n':'Recommendations',
 'Discussion of Recommendations: Opioid Therapy for Acute Pain: Discussion':'Recommendations',
 'Discussion of Recommendations: Risk Mitigation: Discussion: Prescribing of Naloxone Rescue and Accompanying Education':'Recommendations',
 'Discussion of Recommendations: Opioid Therapy for Acute Pain: Recommendation':'Recommendations',
 'Recommendations':'Recommendations',
 'About this Clinical Practice Guideline: Shared Decision Making': 'Specifications',
 'Discussion of Recommendations: Initiation and Continuation of Opioids: Recommendation \n':'Recommendations',
 'Introduction' : 'Introduction',
 'Discussion of Recommendations: Risk Mitigation: Discussion: Other Risk Mitigation Strategies':'Recommendations',
 'Paradigm Shift in Pain and Its Treatment':'Background',
 'Discussion of Recommendations: Type, Dose, Duration, Follow-up, and Taper of Opioids: Recommendations':'Recommendations',
 'Epidemiology and Impact: General Population':'Background',
 'Discussion of Recommendations: Risk Mitigation: Discussion':'Recommendations',
 'About this Clinical Practice Guideline: Methods': 'Specifications',
 'Opioid Epidemic':'Background',
 'Discussion of Recommendations: Initiation and Continuation of Opioids: Discussion \n':'Recommendations',
 'Module C: Tapering or Discontinuation of Opioid Therapy': 'Algorithm',
 'Discussion of Recommendations: Type, Dose, Duration, Follow-up, and Taper of Opioids: Discussion':'Recommendations',
 'Discussion of Recommendations: Risk Mitigation: Discussion: State Prescription Drug Monitoring Programs':'Recommendations',
 'Discussion of Recommendations: Initiation and Continuation of Opioids: Recommendations \n':'Recommendations',
 'How to Use This Clinical Practice Guideline': 'Introduction',
 'Chronic Pain and Co-occurring Conditions ':'Background',
 'Discussion of Recommendations: Risk Mitigation: Discussion: Patients at High Risk for Opioid Use Disorder':'Recommendations',
 'Module A: Determination of Appropriateness for Opioid Therapy': 'Algorithm',
 'Risk Factors for Adverse Outcomes of Opioid Therapy ':'Background',
 'Taxonomy':'Background',
 'About this Clinical Practice Guideline': 'Specifications',
 'About this Clinical Practice Guideline: Patient-centered Care': 'Specifications',
 'Pain Management Opioid Taper Decision Tool': 'Specifications',
 'Prioritizing Safe Opioid Prescribing Practices and Use ':'Background',
 'Discussion of Recommendations: Initiation and Continuation of Opioids: Discussion':'Recommendations',
 'Significant Risk Factors':'Background',
 'About this Clinical Practice Guideline: Clinical Decision Support Tools': 'Specifications',
 'Qualifying Statements':'Introduction',
 'Module D: Patients Currently on Opioid Therapy': 'Algorithm',
 'Mental health disorders':'Background',
 'Discussion of Recommendations: Risk Mitigation: Discussion: Urine Drug Testing and Confirmatory Testing':'Recommendations',
 'Discussion of Recommendations: Initiation and Continuation of Opioids: Recommendation':'Recommendations',
 'About this Clinical Practice Guideline: Scope of this Clinical Practice Guideline': 'Specifications',
 'Algorithm': 'Algorithm',
 'Module B: Treatment with Opioid Therapy': 'Algorithm',
 'About this Clinical Practice Guideline: Highlighted Features of this Clinical Practice Guideline': 'Specifications',
 'Epidemiology and Impact: VA/DoD Population ': 'Background'}


df['title_new'] = df['title'].map(new_title)

squad_entries=[]
for i in tqdm(range(len(df))):
    id_num      = i
    context     = df['context'].values[i]
    question    = df['question'].values[i]
    answer_text = df['answer'].values[i]
    start       = df['start_index'].values[i]
    title       = df['title_new'].values[i]
    
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
    
    full_entry = {
                  "paragraphs":new_entry,
                  "title":title
                 }
    squad_entries.append(full_entry)
    
    
def entry_to_json(squad_entries,title=None,version=None):
    squad_json={
               "data": squad_entries,
               "version":version
               }
    return squad_json


otquad_json = entry_to_json(squad_entries,title=None,version=1.0)


json_file_name = "cpgQA-v1.0.json"

with open(data_path/json_file_name, "w") as writer:
    writer.write(json.dumps(otquad_json, indent=4) + "\n")
