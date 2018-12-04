# -*- coding: utf-8 -*
from fyz_hmm_wordseg import load_dict
from fyz_hmm_wordseg import cut

Sp_matrix, Tp_matrix, Op_matrix ,State_list= load_dict()

def process_data_file(line):
    line = cut(line,Sp_matrix, Tp_matrix, Op_matrix,State_list)
    return line

def main():
    line = ""
    print(process_data_file(line))


if __name__ == "__main__":
    main()