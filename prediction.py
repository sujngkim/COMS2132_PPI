import pickle 
import torch
import data_processing_2 as dt
import deep_learning_model as m

def processing_input():
    with open('processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
        input_type = input("Would you like to input protein id or the amino acid sequence?: ")
        wrong_input_type = False
        while not wrong_input_type:
            if input_type == "protein id":
                wrong_input_type = True
                prot_1 = input("first protein id: ")
                invalid_protein_1 = False
                invalid_protein_2 = False
                while not invalid_protein_1:
                    try:
                        prot_1_seq = data['prots'][prot_1]
                        invalid_protein_1 = True
                    except KeyError:
                        print("first protein id cannot be found, please input a different protein id")
                        prot_1 = input("first protein id: ")
                prot_2 = input("second protein id: ")        
                while not invalid_protein_2:
                    try:
                        prot_2_seq = data['prots'][prot_2]
                        invalid_protein_2 = True
                    except KeyError:
                        print("second protein id cannot be found, please input a different protein id")
                        prot_2 = input("second protein id: ")
                return prot_1_seq, prot_2_seq # tensors

            if input_type == "amino acid sequence":
                wrong_input_type = True
                prot_1 = input("first amino acid sequence: ").upper()
                invalid_seq_1 = False
                invalid_seq_2 = False
                while not invalid_seq_1:
                    try:
                        prot_1_seq = [dt.aa_to_int()[a] for a in prot_1]
                        if len(prot_1_seq) <= 1024:
                            prot_1_seq = torch.tensor(prot_1_seq + [0]*(1024-len(prot_1_seq)), dtype=torch.long)
                        else:
                            prot_1_seq = torch.tensor(prot_1_seq[:512] + prot_1_seq[len(prot_1_seq)-512:], dtype=torch.long)
                        invalid_seq_1 = True
                    except KeyError:
                        print("first amino acid sequence is invalid, please enter characters in [ACDEFGHIKLMNPQRSTVWY]")
                        prot_1 = input("first amino acid sequence: ").upper()
                prot_2 = input("second amino acid sequence: ").upper()
                while not invalid_seq_2:
                    try:
                        prot_2_seq = [dt.aa_to_int()[a] for a in prot_2]
                        if len(prot_2_seq) <= 1024:
                            prot_2_seq = torch.tensor(prot_2_seq + [0]*(1024-len(prot_2_seq)), dtype=torch.long)
                        else:
                            prot_2_seq = torch.tensor(prot_2_seq[:512] + prot_2_seq[len(prot_2_seq)-512:], dtype=torch.long)
                        invalid_seq_2 = True
                    except KeyError:
                        print("first amino acid sequence is invalid, please enter characters in [ACDEFGHIKLMNPQRSTVWY]")
                        prot_2 = input("first amino acid sequence: ").upper()
                return prot_1_seq, prot_2_seq # tensors
            else:
                input_type = input("Please type either protein id or the amino acid sequence: ")
        
def reinitializing_model():
    trained_model = m.SiamesePPI() 

    # 1. Load the weights from the file
    # Use map_location=device to handle switching between GPU and CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.load_state_dict(torch.load("siamese_ppi_model.pth", map_location=device))

    # 2. Set to evaluation mode
    trained_model.to(device)
    trained_model.eval()
    return trained_model, device

def _string_score_feature(combined_score: float, device: torch.device) -> torch.Tensor:
    if combined_score <= 0:
        w = 0.0
    else:
        w = 1.0
    return torch.tensor([[w]], dtype=torch.float32, device=device)

def predict_interaction(prot_a_seq, prot_b_seq, model, device, combined_score: float):

    tensor_a = prot_a_seq.unsqueeze(0).to(device)
    tensor_b = prot_b_seq.unsqueeze(0).to(device)
    score_t = _string_score_feature(combined_score, device)
    with torch.no_grad():
        p_ab = model(tensor_a, tensor_b, score_t).item()
        p_ba = model(tensor_b, tensor_a, score_t).item()
        probability = max(p_ab, p_ba) * 100
    return probability

def main():
    prot_a_seq, prot_b_seq = processing_input()
    combined_score = 924
    model, device = reinitializing_model()
    probability = predict_interaction(prot_a_seq, prot_b_seq, model, device, combined_score)
    print("the two proteins have a {:.2f}% chance of interacting".format(probability))

if __name__ == "__main__":
    main()
