"""
Creates per-patient embedding vectors from model, maps to patient IDs,
builds reference indices for ANN search, and finds neighbors
"""
import argparse
import os
import pickle

import numpy as np
from annoy import AnnoyIndex


def get_per_patient_med2vec_representations(work_dir, prefix, med2vec_model):
    """Generates Per-Patient Mappings to Med2Vec Format Input"""
    with open(os.path.join(work_dir, prefix + '_med2vec_vectors.pkl'), 'rb') as file:
        raw_x = pickle.load(file)
    with open(os.path.join(work_dir, prefix + '_patient_indexes.pkl'), 'rb') as file_p:
        patient_ids = pickle.load(file_p)
    X = np.zeros((len(raw_x), 19266))  # TODO do not hardcode this/load dynamically
    for idx, seq in enumerate(raw_x):
        if not seq[0] == -1:
            X[idx][seq] = 1
    print(X.shape)
    for i in range(1, 100):
        print(X[i].sum())
    # read in the weights and bias & implement the equation
    params = np.load(med2vec_model)
    W_emb = np.array(params['W_emb'])
    b_emb = np.array(params['b_emb'])
    W_hidden = np.array(params['W_hidden'])
    b_hidden = np.array(params['b_hidden'])

    emb = np.maximum((np.dot(X, W_emb) + b_emb), 0)
    visit = np.maximum((np.dot(emb, W_hidden) + b_hidden), 0)

    patient_rep = []
    temp = []
    patient_id_list = []
    p_idx = 0
    for idx, seq in enumerate(raw_x):
        if seq[0] == -1:
            pr = np.array(temp).sum(0) / len(temp)
            patient_rep.append(pr)
            patient_id_list.append(patient_ids[p_idx])
            p_idx += 1
            temp = []
        else:
            temp.append(visit[idx])

    print(len(patient_rep))
    print(len(patient_rep[0]))

    patient_rep_dict = dict(zip(patient_id_list, patient_rep))
    # store the output in a dict with patient id as key and list of visit representation as value.
    with open(os.path.join(work_dir, prefix + "_patient_emb.pkl"), "wb") as file_r:
        pickle.dump(patient_rep, file_r)

    with open(os.path.join(work_dir, prefix + "_patient_emb_withid.pkl"), "wb") as file_d:
        pickle.dump(patient_rep_dict, file_d)


distance_type = ["angular", "euclidean", "manhattan", "hamming", "dot"]


def build_annoy_tree(work_dir, data, vec_len, dtype, tree_num=50):
    t = AnnoyIndex(vec_len, dtype)
    for idx, patient in enumerate(data):
        t.add_item(idx, patient)

    t.build(tree_num)  # 20 trees
    t.save(os.path.join(work_dir, 'reference_' + dtype + '.index'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('working_dir', type=str)
    parser.add_argument('embeddings_file', type=str)
    args = parser.parse_args()

    # Convert reference and case to Med2Vec format input
    get_per_patient_med2vec_representations(args.working_dir, 'reference', args.embeddings_file)
    get_per_patient_med2vec_representations(args.working_dir, 'case', args.embeddings_file)

    # Build the Annoy Indices for reference dataset
    with open(os.path.join(args.working_dir, "reference_patient_emb.pkl"), "rb") as emb_pkl:
        data = pickle.load(emb_pkl)
    vec_len = np.array(data).shape[1]
    print('Annoy Index Vector Length:', vec_len)
    for tree_type in distance_type:
        build_annoy_tree(args.working_dir, data, vec_len, tree_type)

    # Now search for similar patients
    with open(os.path.join(args.working_dir, "case_patient_indexes.pkl"), "rb") as case_patid_pkl:
        case_patients = pickle.load(case_patid_pkl)
    with open(os.path.join(args.working_dir, "reference_patient_indexes.pkl"), "rb") as reference_patid_pkl:
        reference_patients = pickle.load(reference_patid_pkl)
    with open(os.path.join(args.working_dir, "case_patient_emb.pkl"), "rb") as case_emb_pkl:
        case_visits = pickle.load(case_emb_pkl)

    top_dict = {}
    for i in range(len(case_patients)):
        patient = case_patients[i]
        visit = case_visits[i]
        all_list = []
        for dtype in distance_type:
            u = AnnoyIndex(vec_len, dtype)
            u.load(os.path.join(args.working_dir, 'reference_' + dtype + '.index'))  # super fast, will just mmap the file
            l = u.get_nns_by_vector(visit, 100)  # will find the 100 nearest neighbor
            l_id = [reference_patients[index] for index in l]
            all_list = list(set(all_list + l_id))
        top_dict[patient] = all_list

    with open(os.path.join(args.working_dir, 'top100_neighbors.pkl'), 'wb') as t100:
        pickle.dump(top_dict, t100)
