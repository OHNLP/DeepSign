import argparse
import os
import pickle

import numpy as np
import pandas as pd
from dateutil import parser as ps


def gen_sparse_vectors(pat_ids, raw_data, include_patients):
    pat_id_to_sparse_vectors = {}
    for i in range(len(pat_ids)):
        if pat_ids[i] not in include_patients:
            continue
        arr = np.zeros(vocab_size)
        for idx in raw_data[i]:
            arr[idx] = 1
        pat_id_to_sparse_vectors[pat_ids[i]] = arr
    return pat_id_to_sparse_vectors


def calc_jlh(case_counts, reference_counts, num_aberrant, num_reference):
    jlh_score_arr = np.zeros(vocab_size)
    for i in range(vocab_size):
        if case_counts[i] < num_aberrant * .03 and i != 0:
            continue
        prop_aberrant = case_counts[i] / num_aberrant
        prop_reference = reference_counts[i] / num_reference
        jlh_score_arr[i] = prop_aberrant / prop_reference * (prop_aberrant - prop_reference)
    return jlh_score_arr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('working_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('case_dates', type=str)
    parser.add_argument('--concurrent_grace_period', type=int, default=30)
    args = parser.parse_args()

    # Create output directory if it does not yet exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # First, generate datasets for comparison by converting to sparse indices
    case_dates = pd.read_csv(args.case_dates, header=None)
    case_dates.columns = ['PATIENT_ID', 'CASE_DATE']
    with open(os.path.join(args.working_dir, 'vocab_indexes.pkl'), 'rb') as f:
        vocab = pickle.load(f)
    with open(os.path.join(args.working_dir, 'case_dataset_raw.pkl'), 'rb') as f:
        case_dataset = pickle.load(f)
    with open(os.path.join(args.working_dir, 'case_patient_indexes.pkl'), 'rb') as f:
        case_pat_ids = pickle.load(f)
    with open(os.path.join(args.working_dir, 'aberrant_case_patients.pkl'), 'rb') as f:
        aberrant_patients = pickle.load(f)
    with open(os.path.join(args.working_dir, 'reference_patient_vectors.pkl'), 'rb') as f:
        reference_raw_data = pickle.load(f)
    with open(os.path.join(args.working_dir, 'reference_patient_indexes.pkl'), 'rb') as f:
        reference_pat_ids = pickle.load(f)
    with open(os.path.join(args.working_dir, 'top100_neighbors.pkl'), 'rb') as f:
        neighbors = pickle.load(f)

    vocab_size = max(vocab.values()) + 1
    # - Generate pre/concurrent/post datasets and convert to sparse vectors
    # -- First categorize pre/concurrent/post
    def map_type(diagnosis_date, case_date):
        # First convert to proper date type so we can do concurrent with +/- windows
        diagnosis_dtm = ps.parse(str(diagnosis_date))
        case_dtm = ps.parse(str(case_date))
        delta = diagnosis_dtm - case_dtm
        if args.concurrent_grace_period >= delta.days >= args.concurrent_grace_period * -1:
            return 'CONCURRENT'
        if delta.days < args.concurrent_grace_period * -1:
            return 'BEFORE'
        else:
            return 'AFTER'
    case_dataset = case_dataset.set_index('PATIENT_ID').join(case_dates.set_index('PATIENT_ID'))
    case_dataset['ENTRY_TYPE'] = case_dataset.apply(lambda d: map_type(d['DIAGNOSIS_DATE'], d['CASE_DATE']), axis=1)
    case_before = case_dataset[case_dataset['ENTRY_TYPE'] == 'BEFORE']
    case_after = case_dataset[case_dataset['ENTRY_TYPE'] == 'AFTER']
    case_concurrent = case_dataset[case_dataset['ENTRY_TYPE'] == 'CONCURRENT']
    case_all = case_dataset
    # -- Now generate equivalent sparse datasets for each

    def groupByPatient(df):
        return df.drop_duplicates().groupby('PATIENT_ID')['DIAGNOSIS_CODE'].agg(list)
    case_before_series = groupByPatient(case_before)
    case_concurrent_series = groupByPatient(case_concurrent)
    case_after_series = groupByPatient(case_after)
    case_all_series = groupByPatient(case_all)
    case_before_dict = {}
    case_concurrent_dict = {}
    case_after_dict = {}
    case_all_dict = {}

    def map_series_to_sparse_vectors(src_series, target_dict, pat_filter):
        for pat_id in pat_filter:
            if pat_id not in src_series:
                continue
            outArr = np.zeros(vocab_size)
            for icd in src_series[pat_id]:
                outArr[vocab[icd]] = 1
            target_dict[pat_id] = outArr

    map_series_to_sparse_vectors(case_before_series, case_before_dict, case_pat_ids)
    map_series_to_sparse_vectors(case_concurrent_series, case_concurrent_dict, case_pat_ids)
    map_series_to_sparse_vectors(case_after_series, case_after_dict, case_pat_ids)
    map_series_to_sparse_vectors(case_all_series, case_all_dict, case_pat_ids)

    # - Generate Reference Patient Set from Case Patient Neighbors
    ref_patient_ids = []
    for case_patid in aberrant_patients:
        ref_patient_ids += neighbors[case_patid]
    ref_patient_ids = set(ref_patient_ids)
    ref_dict = gen_sparse_vectors(reference_pat_ids, reference_raw_data, ref_patient_ids)

    # Now, get JLH scores
    aberrant_cohort_size = float(len(aberrant_patients))
    reference_cohort_size = float(len(ref_dict))

    # - Sum within each dict to get number of patients with each index
    def get_case_totals(input_case_dict):
        case_totals = np.zeros(vocab_size)
        for vec in input_case_dict.values():
            case_totals = case_totals + vec
        return case_totals

    pre_case_totals = get_case_totals(case_before_dict)
    concurrent_case_totals = get_case_totals(case_concurrent_dict)
    post_case_totals = get_case_totals(case_after_dict)
    all_case_totals = get_case_totals(case_all_dict)

    reference_totals = np.zeros(vocab_size)
    for vec in ref_dict.values():
        reference_totals = reference_totals + vec

    # Invert vocab for lookup
    vocab_idx = {}
    for icd in vocab:
        vocab_idx[vocab[icd]] = icd

    def get_output(case_totals, ref_totals, aberrant_size, reference_size, output_prefix):
        jlh = calc_jlh(case_totals, ref_totals, aberrant_size, reference_size)
        # Now get top 100 indices and map to ICD codes
        indices = range(vocab_size)
        sorted_indices = sorted(list(zip(jlh, indices)), reverse=True)
        top_score_index = sorted_indices[1:]  # First will always be 0 index with nan
        R_idx = [sl[1] for sl in top_score_index]
        top_score = [sl[0] for sl in top_score_index]
        top_scores_mapped = [(vocab_idx[item[1]], item[0], case_totals[item[1]], reference_totals[item[1]]) for item in top_score_index]
        with open(os.path.join(args.output_dir, output_prefix + '_jlh.txt'), 'wb') as op:
            for score in top_scores_mapped:
                op.write(str(score[0]).encode("utf-8"))
                op.write(b",")
                op.write(str(score[1]).encode("utf-8"))
                op.write(b",")
                op.write(str(score[2]).encode("utf-8"))
                op.write(b",")
                op.write(str(score[3]).encode("utf-8"))
                op.write(b"\r\n")

    get_output(pre_case_totals, reference_totals, aberrant_cohort_size, reference_cohort_size, "pre")
    get_output(concurrent_case_totals, reference_totals, aberrant_cohort_size, reference_cohort_size, "concurrent")
    get_output(post_case_totals, reference_totals, aberrant_cohort_size, reference_cohort_size, "post")
    get_output(all_case_totals, reference_totals, aberrant_cohort_size, reference_cohort_size, "all")

