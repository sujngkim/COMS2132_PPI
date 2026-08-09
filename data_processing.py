from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from pathlib import Path
import numpy as np
import random

def file_path(file_name):
    
    ''' returns the path of the file'''

    current_dir = Path(".") 
    data_file = current_dir /file_name
    return data_file

def read_score_file(file):
    with open(file, 'r') as f:
        f.readline()
        all_interactions = {}
        for line in f:
            li = line.split()
            prot_1 = li[0].split(".")[1]
            prot_2 = li[1].split(".")[1]
            score = int(li[2])
            all_interactions[(prot_1, prot_2)] = score
            all_interactions[(prot_2, prot_1)] = score
    return all_interactions

def valid_interactions(dict):

    '''reads in the protein interaction score file 
    and only adds the interactions with a score higher than 800
    ''' 
    prot_interactions = {}
    for (p, s) in dict.items():
        if s > 800:
            prot_interactions[p] = s
    return prot_interactions # {(protein1 id, protein2 id) : confidence score}

def negative_set(id_dict, score_dict):

    '''using the existing score dictionary, create a dictionary of protein interactions that 
    are not in the dictionary, set all the weight as the avg of the positive dataset
    '''

    score_sum = 0
    id_list = list(id_dict)
    for i in score_dict:
        score_sum += score_dict[i]
    avg_score = score_sum / len(score_dict)
    neg_data = {}
    while len(neg_data) < len(score_dict):
        p1 = random.choice(id_list)
        p2 = random.choice(id_list)
        if (p1, p2) not in score_dict and (p1, p2) not in neg_data: 
            neg_data[(p1, p2)] = avg_score
            neg_data[(p2, p1)] = avg_score
    return neg_data

def read_protein_id(file):

    ''' reads in the file and makes a dictionary
    where the key is the protein id and the value is the AA sequence
    '''

    protein_id = {}
    for record in SeqIO.parse(file, "fasta"):
        id = record.id.replace('4932.', '')
        protein_id[id] = str(record.seq)
    return protein_id # {protein id : amino acid seq}

def aa_stats(dict):

    ''' the minimum and maximum length of the amino acid sequence'''

    # max = 0
    # for seq in dict.values():
    #     if len(seq) > max:
    #         max = len(seq)
    # min = max
    # for seq in dict.values():
    #     if len(seq) < min:
    #         min = len(seq)
    # return max, min
    sum = 0
    for seq in dict.values():
        sum += len(seq)
    return sum / len(dict)

def extract_physicochemical_features(seq):

    ''' takes in a single amino acid sequence as a parameter,
    returns a dictionary of physicochemical features
    '''

    analysed_seq = ProteinAnalysis(seq)
    features =  np.array([analysed_seq.molecular_weight(),analysed_seq.aromaticity(),analysed_seq.instability_index(),\
                          analysed_seq.isoelectric_point(),analysed_seq.gravy()])
    return features # 1D array of features

def id_to_feature(dict):
    protein_features = {} #{id: feature array}
    for (id, aa) in dict.items():
        protein_features[id] = extract_physicochemical_features(aa)
    return protein_features

def array_conversion(interaction_dict, feature_dict):

    ''' constructs a numpy array for each interaction
    iterate through the interaction dictionary and concatenate the features of each protein
    reverse the order and repeat for symmetry
    construct a second numpy array for the confidence score
    '''
    
    # make an 2D array by iterating through the interaction dictionary
    # concatenate the 1d arrays for each aa 

    li_interaction_dict = list(interaction_dict)
    weight_array = np.zeros((len(interaction_dict),))
    interaction_array = np.zeros((len(interaction_dict), 10))
    for i in range(len(interaction_dict)):
        interaction_array_row = np.concatenate((feature_dict[li_interaction_dict[i][0]], feature_dict[li_interaction_dict[i][1]]))
        interaction_array[i] = interaction_array_row
        weight_array[i] = interaction_dict[li_interaction_dict[i]]
    
    return interaction_array, weight_array

if __name__ == "__main__":
    pass
    # valid_interaction = valid_interactions(all_interaction)
    # features = id_to_feature(protein_id)   
    # features_array, weight_array = array_conversion(valid_interaction, features)