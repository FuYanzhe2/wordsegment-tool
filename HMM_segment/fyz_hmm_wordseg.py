# -*- coding: utf-8 -*

import json
import re
import os
import math

#data_root_path = r'/home/fyz/nlp/wordseg/fyz_hmm_wordseg/data/seg_dataset_product1/'
Corpus_Dataset = "trainCorpus.txt"
#Corpus_Dataset = "/home/fyz/nlp/wordseg/fyz_hmm_wordseg/training_data/seg_dataset/598b4777077a8b43d4284102.txt"
State_set = ['B','M','E','S']
Transprob_dict = {}
Observedpro_dict = {}
Startpro_dict = {}
Count_dict = {}
State_list = {}
MIN_FLOAT = -3.14e100
PROB_START = "prob_start"   #初始状态概率
PROB_OBSER = "prob_obser"     #发射概率
PROB_TRANS = "prob_trans"   #转移概率
STATE_WORD = "state_list"
MIN_INF = float("-inf")
Traning = False

def load_dict():
    fs = open("prob_start.json",'r')
    ft = open("prob_trans.json", 'r')
    fo = open("prob_obser.json", 'r')
    fstate_list = open("state_list.json", 'r')
    Sp_matrix = json.load(fs)
    Tp_matrix = json.load(ft)
    Op_matrix = json.load(fo)
    State_list = json.load(fstate_list)
    return Sp_matrix, Tp_matrix, Op_matrix,State_list

def viterbi(obs, states, start_p, trans_p, emit_p,State_list):
    V = [{}]
    path = {}
    mem_path = [{}]
    for y in State_list.get(obs[0],states):
        V[0][y] = start_p[y] +emit_p[y].get(obs[0],MIN_FLOAT)
        path[y] = [y]
        mem_path[0][y] = ''
    for t in range(1,len(obs)):
        V.append({})
        mem_path.append({})
        newpath = {}
        prev_states = [
            x for x in mem_path[t - 1].keys() if len(trans_p[x]) > 0]
        prev_states_expect_next = set(
            (y for x in prev_states for y in trans_p[x].keys()))
        obs_states = set(
            State_list.get(obs[t], states)) & prev_states_expect_next
        if not obs_states:
            obs_states = prev_states_expect_next if prev_states_expect_next else states

        for y in obs_states:   #从y0 -> y状态的递归
            a =emit_p[y].get(obs[t], MIN_FLOAT)
            (prob, state) = max([(V[t-1][y0] +trans_p[y0].get(y,MIN_INF) +emit_p[y].get(obs[t],MIN_FLOAT) ,y0)
                                 for y0 in prev_states])

            V[t][y] =prob
            newpath[y] = path[state] + [y]
            mem_path[t][y] = state
        path = newpath  #记录状态序列
    last = [(V[-1][y], y) for y in mem_path[-1].keys()]
    # if len(last)==0:
    #     print obs
    prob, state = max(last)
    route = [None] * len(obs)
    i = len(obs) - 1
    while i >= 0:
        route[i] = state
        state = mem_path[i][state]
        i -= 1
    #(prob, state) = max([(V[len(obs) - 1][y], y) for y in states])  #在最后一个位置，以y状态为末尾的状态序列的最大概率
    return (prob,route)  #返回概率和状态序列

def cut(sentence,prob_start, prob_trans, prob_emit,State_list):
    out_put = ""
    prob, pos_list =  viterbi(sentence,('B','M','E','S'), prob_start, prob_trans, prob_emit,State_list)
    for i, v in enumerate(pos_list) :

        if v == 'E' or v == 'S' :
            out_put =  out_put+sentence[i]
            out_put =  out_put + " "
        else:
            out_put = out_put + sentence[i]
    return out_put

def init_dict():
    for state in State_set:
        Count_dict[state] = 0.0
        Startpro_dict[state] = 0.0
        Transprob_dict[state] = {}
        Observedpro_dict[state] = {}
        for state_ in State_set:
            Transprob_dict[state][state_] = 0.0

def get_word_state(words):
    output_list = []
    if len(words) == 1:
        output_list.append('S')
    elif len(words) == 2:
        output_list = ['B','E']
    else:
        middle_list = ['M']*(len(words)-2)
        output_list.append('B')
        output_list.extend(middle_list)
        output_list.append('E')
    return output_list

def get_pro_matrix(Sp_dict, Tp_dict, Op_dict, Count_dict ,Count_start,State_list):
    start_fp = open(PROB_START+".json",'w')
    obser_fp = open(PROB_OBSER+".json",'w')
    trans_fp = open(PROB_TRANS+".json",'w')
    state_list = open(STATE_WORD+".json",'w')
    for key in State_set:
        if Sp_dict[key] <= 0:
            Sp_dict[key] = MIN_FLOAT
        else:
            Sp_dict[key] = Sp_dict[key]*1.0/Count_start
            Sp_dict[key] = math.log(Sp_dict[key])

    #start_fp.write(str(Sp_dict))
    json.dump(Sp_dict,start_fp,ensure_ascii=False)
    json.dump(State_list, state_list, ensure_ascii=True)
    for key in Tp_dict :
        for key_ in Tp_dict[key]:
            if(Count_dict[key]==0):
                Tp_dict[key][key_]=0
            else:
                Tp_dict[key][key_] =Tp_dict[key][key_]*1.0/Count_dict[key]

            if Tp_dict[key][key_]<=0:
                Tp_dict[key][key_] = MIN_INF
            else:
                Tp_dict[key][key_] = math.log(Tp_dict[key][key_])
    Tp_dict['E'].pop('E')
    Tp_dict['E'].pop('M')
    Tp_dict['B'].pop('S')
    Tp_dict['B'].pop('B')
    Tp_dict['S'].pop('E')
    Tp_dict['S'].pop('M')
    Tp_dict['M'].pop('S')
    Tp_dict['M'].pop('B')

    #trans_fp.write(str(Tp_dict))
    json.dump(Tp_dict, trans_fp, ensure_ascii=False)

    for key in Op_dict :
        for word in Op_dict[key]:
            Op_dict[key][word] = math.log(Op_dict[key][word]*1.0/Count_dict[key])
            #if Op_dict[key][word]==0:
                #Op_dict[key][word] = MIN_INF

        if key == 'S':
            Op_dict[key]['UNK'] = math.log(0.1)
        else:
            Op_dict[key]['UNK'] = math.log(0.3)

    #obser_fp.write(str(Op_dict))
    json.dump(Op_dict, obser_fp, ensure_ascii=True)
    return Sp_dict,Tp_dict,Op_dict,State_list

def count_dict_key(corpus_file):
    num_file = 0
    Count_start = 0
    num_file = num_file + 1
    if(num_file%1000==0):
        print (num_file)
    with open(corpus_file, 'r') as f:
        #words_state_lists = []
        for line in f.readlines():
            line = line.strip().split()
            words_state_lists = []
            words_list = []
            for words in line:
                words_state_lists.extend(get_word_state(words))
                for word in words:
                    words_list.append(word)
            assert len(words_state_lists) == len(words_list), "length error"

            for i in range(len(words_state_lists)):
                if words_list[i] in State_list:
                    if words_state_lists[i] not in State_list[words_list[i]] :
                        State_list[words_list[i]].append(words_state_lists[i])
                else:
                    State_list[words_list[i]] = []
                    State_list[words_list[i]].append(words_state_lists[i])

            #assert len(words_state_lists) == len(words_list),"length error"
            #for i in range(len(words_state_lists)):
                if words_list[i] not in Observedpro_dict[words_state_lists[i]]:
                    Observedpro_dict[words_state_lists[i]][words_list[i]] = 1.0
                else:
                    Observedpro_dict[words_state_lists[i]][words_list[i]] += 1.0

                if i==0 :
                    Count_start+=1
                    Startpro_dict[words_state_lists[i]] +=1.0
                    Count_dict[words_state_lists[i]] +=1.0
                else:
                    Transprob_dict[words_state_lists[i-1]][words_state_lists[i]] +=1.0
                    Count_dict[words_state_lists[i]] += 1.0
    print (Startpro_dict)
    print (Count_start)
    return Startpro_dict,Transprob_dict,Observedpro_dict,Count_dict,Count_start,State_list

def get_file_name(data_root_path, file_type):
    file_name = []
    for root, dir, files in os.walk(data_root_path):
        for file in files:
            if file_type in file:
                file_name.append(file)
    return file_name

def main():
    test_str = u"四项基本原则是我们立国的根本"
    test_str = test_str.strip()

    if Traning :
        init_dict()
        Sp_dict, Tp_dict, Op_dict, Count_dict, Count_start ,State_list= count_dict_key(Corpus_Dataset)
        Sp_matrix, Tp_matrix, Op_matrix ,State_list= get_pro_matrix(Sp_dict, Tp_dict, Op_dict, Count_dict ,Count_start,State_list)
    else:
        Sp_matrix, Tp_matrix, Op_matrix ,State_list= load_dict()

    out_put= cut(test_str,Sp_matrix, Tp_matrix, Op_matrix,State_list)
    print(test_str)
    print (out_put)


if __name__ == "__main__":
    main()
