import random
import data_processing_2 as dt
import csv

def neg_set(file, id_dict, score_dict):
    with open(file, 'r') as f:
        f.readline()
        all_interactions = []
        for line in f:
            li = line.split()
            all_interactions.append(li[:1])
            all_interactions.append(li[:1:-1])
    score_sum = 0
    id_list = list(id_dict)
    for i in score_dict:
        score_sum += score_dict[i]
    avg_score = score_sum / len(score_dict)
    neg_data = []
    while len(neg_data) < len(score_dict):
        p1 = random.choice(id_list)
        p2 = random.choice(id_list)
        if (p1, p2) not in score_dict and (p1, p2) not in neg_data: 
            neg_data.append([p1, p2, avg_score])
            neg_data.append([p2, p1, avg_score])
    with open("neg_data.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(neg_data)

    return f
def main():
    protein_file = dt.file_path("4932.protein.sequences.v12.0.fa")
    aa_to_int = dt.aa_to_int()
    protein_id = dt.read_protein_id(protein_file, aa_to_int)
    all_interactions_file = dt.file_path("4932.protein.links.v12.0.txt")
    valid_interactions = dt.valid_interactions(all_interactions_file)
    neg_set(all_interactions_file, protein_id, valid_interactions)

if __name__ == "__main__":
    main()
# make valid score file in dp2
# import id dictionary from dp2
# import score dictionary to match the length and weight
# read in the original interaction file
# make a csv file