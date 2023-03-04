import os
import openai
import json
import numpy as np
from numpy.linalg import norm
import re
from time import time,sleep
from uuid import uuid4
from fastapi import FastAPI, HTTPException
import pdb


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)


def save_json(filepath, payload):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)


def gpt3_embedding(content, engine='text-embedding-ada-002'):
    content = content.encode(encoding='ASCII',errors='ignore').decode()
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


def similarity(v1, v2):
    # based upon https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
    return np.dot(v1, v2)/(norm(v1)*norm(v2))  # return cosine similarity


def fetch_memories(vector, logs, count):
    scores = list()
    for i in logs:
        if vector == i['vector']:
            # skip this one because it is the same message
            continue
        score = similarity(i['vector'], vector)
        i['score'] = score
        scores.append(i)
    ordered = sorted(scores, key=lambda d: d['score'], reverse=True)
    # TODO - pick more memories temporally nearby the top most relevant memories
    try:
        ordered = ordered[0:count]
        return ordered
    except:
        return ordered


def load_convo():
    files = os.listdir('interview_logs')
    files = [i for i in files if '.json' in i]  # filter out any non-JSON files
    result = list()
    for file in files:
        data = load_json('interview_logs/%s' % file)
        result.append(data)
    ordered = sorted(result, key=lambda d: d['time'], reverse=False)  # sort them all chronologically
    return result

def fetch_research(vector, research_folder, count):
    research_files = os.listdir(research_folder)
    research_logs = []
    for file in research_files:
        content = open_file(os.path.join(research_folder, file))
        research_vector = gpt3_embedding(content)
        score = similarity(vector, research_vector)
        research_logs.append({'file': file, 'vector': research_vector, 'score': score})
    ordered = sorted(research_logs, key=lambda d: d['score'], reverse=True)[:count]
    return ordered


def summarize_memories_and_research(memories, research):  # summarize a block of memories into one payload
    memories = sorted(memories, key=lambda d: d['time'], reverse=False)  # sort them chronologically
    block = ''
    for mem in memories:
        block += '%s: %s\n\n' % (mem['speaker'], mem['message'])
    
    research_folder = 'dan_info'
    # Combine the research into a single block
    relevant_research = research
    research_text = ''
    for research in relevant_research:
        research_text += open_file(os.path.join(research_folder, research['file'])) + '\n\n'
    
    # Combine the block of memories and research into a single prompt
    block = block.strip()
    research_text = research_text.strip()
    prompt = open_file('prompt_notes.txt').replace('<<INPUT>>', block + research_text)
    # TODO - do this in the background over time to handle huge amounts of memories
    notes = gpt3_completion(prompt)
    return notes


def get_last_messages(conversation, limit):
    try:
        short = conversation[-limit:]
    except:
        short = conversation
    output = ''
    for i in short:
        output += '%s: %s\n\n' % (i['speaker'], i['message'])
    output = output.strip()
    return output


def gpt3_completion(prompt, engine='text-davinci-003', temp=0.0, top_p=1.0, tokens=400, freq_pen=0.0, pres_pen=0.0, stop=['USER:', 'BENJI:']):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response['choices'][0]['text'].strip()
            text = re.sub('[\r\n]+', '\n', text)
            text = re.sub('[\t ]+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            if not os.path.exists('interview_gpt3_logs'):
                os.makedirs('interview_gpt3_logs')
            save_file('interview_gpt3_logs/%s' % filename, prompt + '\n\n==========\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)




if __name__ == '__main__':
    openai.api_key = open_file('openaiapikey.txt')
    while(True):
        #### get user input, save it, vectorize it, etc
        a = input('\n\nUSER: ')
        vector = gpt3_embedding(a)
        info = {'speaker': 'USER', 'time': time(), 'vector': vector, 'message': a, 'uuid': str(uuid4())}
        filename = 'log_%s_USER.json' % time()
        save_json('interview_logs/%s' % filename, info)


        #### load conversation
        conversation = load_convo()

        #### compose corpus (fetch memories and research)
        memories = fetch_memories(vector, conversation, 10)  # pull episodic memories
        # TODO - fetch declarative memories (facts, wikis, KB, company data, internet, etc)
        research = fetch_research(vector, 'dan_info', 3)  # pull research
        notes = summarize_memories_and_research(memories, research)
        recent = get_last_messages(conversation, 4)
        # print("recent")
        # print(recent)
        # pdb.set_trace()
        prompt = open_file('prompt_interviewer.txt').replace('<<NOTES>>', notes).replace('<<CONVERSATION>>', recent).replace('<<MESSAGE>>', a.strip())
        #### generate response, vectorize, save, etc
        output = gpt3_completion(prompt)
        vector = gpt3_embedding(output)
        info = {'speaker': 'BENJI', 'time': time(), 'vector': vector, 'message': output, 'uuid': str(uuid4())}
        filename = 'log_%s_BENJI.json' % time()
        save_json('interview_logs/%s' % filename, info)
        #### print output
        print('\n\nBENJI: %s' % output) 

     
