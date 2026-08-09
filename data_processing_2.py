from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pickle
import torch
import random
import csv

def file_path(file_name):
    
    ''' returns the path of the file'''

    current_dir = Path(".") 
    data_file = current_dir /file_name
    return data_file

# def read_score_file(file):
#     with open(file, 'r') as f:
#         f.readline()
#         all_interactions = {}
#         for line in f:
#             li = line.split()
#             prot_1 = li[0].split(".")[1]
#             prot_2 = li[1].split(".")[1]
#             score = int(li[2])
#             all_interactions[(prot_1, prot_2)] = score
#             all_interactions[(prot_2, prot_1)] = score
#     return all_interactions

def valid_interactions(file):

    '''reads in the protein interaction file 
    and only adds the interactions with a score higher than 800
    ''' 
    with open(file, 'r') as f:
        f.readline()
        prot_interactions = {}
        for line in f:
            li = line.split()
            prot_1 = li[0].split(".")[1]
            prot_2 = li[1].split(".")[1]
            score = int(li[2])
            if score > 800:
                prot_interactions[(prot_1, prot_2)] = score
                prot_interactions[(prot_2, prot_1)] = score
    return prot_interactions # {(protein1 id, protein2 id) : confidence score}

# def negative_set(id_dict, score_dict):

#     '''using the existing score dictionary, create a dictionary of protein interactions that 
#     are not in the dictionary, set all the weight as the avg of the positive dataset
#     '''

#     score_sum = 0
#     id_list = list(id_dict)
#     for i in score_dict:
#         score_sum += score_dict[i]
#     avg_score = score_sum / len(score_dict)
#     neg_data = {}
#     while len(neg_data) < len(score_dict):
#         p1 = random.choice(id_list)
#         p2 = random.choice(id_list)
#         if (p1, p2) not in score_dict and (p1, p2) not in neg_data: 
#             neg_data[(p1, p2)] = avg_score
#             neg_data[(p2, p1)] = avg_score
#     return neg_data

def negative_set(file):
    prot_noninteractions = {}
    with open(file, newline ='') as f:
        neg_set = csv.reader(f)
        for row in neg_set:
            prot_noninteractions[(row[0], row[1])] = float(row[2])
    return prot_noninteractions
    
def aa_to_int():

    ''' make a dictionary where each amino acid corresponds to an integer'''

    aa_int = {}
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    i = 1
    for a in amino_acids:
        aa_int[a] = i
        i += 1
    return aa_int

def read_protein_id(file, dict): 

    ''' reads in the file and makes a dictionary
    where the key is the protein id and the value is the list of AA sequence in integers
    '''

    protein_id = {}
    for record in SeqIO.parse(file, "fasta"):
        id = record.id.replace('4932.', '')
        int_seq = []
        for a in str(record.seq):
            int_seq.append(dict[a])
        protein_id[id] = int_seq
    return protein_id # {protein id : amino acid seq in integer}


def padding_and_truncating_seq(dict):

    ''' get the id to protein sequnce dictionary. 
    if the sequnce is longer than 1024, cut the middle part out.
    If the length is shorter, add 0s to the end of the sequence until the length is 1024'''

    for (id, seq) in dict.items():
        if len(seq) <= 1024:
            dict[id] = torch.tensor(seq + [0]*(1024-len(seq)))
        else:
            dict[id] = torch.tensor(seq[:512] + seq[len(seq)-512:], dtype=torch.long)
    return dict # {id: tensor of amino acid sequnce in integers padded and trucated}

class PPIDataset(Dataset):

    def __init__(self, interaction_dict, sequence_dict, label):

        ''' takes in the interaction dictionary and 
        the processed amino acid sequence dictionary to create a Dataset instance
        '''

        self.interaction_pairs = list(interaction_dict.keys())
        self.interaction_dict = interaction_dict
        self.sequence_dict = sequence_dict
        self.label = label

    def __len__(self):
        return len(self.interaction_pairs)

    def __getitem__(self, idx): # wrap the data in a tensor
        id_a, id_b = self.interaction_pairs[idx]
        seq_a = self.sequence_dict[id_a].detach().clone()
        seq_b = self.sequence_dict[id_b].detach().clone()
        label = torch.tensor(self.label, dtype=torch.float32)
        weight = torch.tensor(float((self.interaction_dict[(id_a, id_b)])))
        return seq_a, seq_b, label, weight
    
def data_pickling():
    protein_id = read_protein_id(file_path("4932.protein.sequences.v12.0.fa"), aa_to_int())
    processed_protein = padding_and_truncating_seq(protein_id)
    valid_interaction = valid_interactions(file_path("4932.protein.links.v12.0.txt"))
    negative_interaction = negative_set(file_path("neg_data.csv"))
    valid_interaction_train = dict(list(valid_interaction.items())[5000:])
    valid_interaction_test = dict(list(valid_interaction.items())[:5000])
    negative_interaction_train = dict(list(negative_interaction.items())[5000:])
    negative_interaction_test = dict(list(negative_interaction.items())[:5000])
   
    with open('processed_data.pkl', 'wb') as f:
        pickle.dump({
            'pos_train': valid_interaction_train,
            'neg_train': negative_interaction_train,
            'pos_test' : valid_interaction_test,
            'neg_test' : negative_interaction_test,
            'prots': processed_protein
        }, f)
    return None
if __name__ == "__main__":
    pass
    # print(negative_set(file_path("neg_data.csv")))
    # data_pickling()


 
    
