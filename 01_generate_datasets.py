"""
Converts ICD-9-CM to ICD-10-CM for consistency across data, and then converts component data to med2vec formats

Expects input data in the format:

[PATIENT_ID, PATIENT_BIRTH_DATE, PATIENT_BIRTH_GENDER, DIAGNOSIS_CODE, DIAGNOSIS_DATE, CODE_SYSTEM, CASE_FLAG]

where CODE_SYSTEM is one of {"ICD9", "ICD10"} and CASE_FLAG is 1 if this patient is a case patient, 0 otherwise

Expected parameters:

* file_path - input file path
* output_dir - output working directory for data
* --train_set_prop - proportion of reference data to use for training. Default .5
* --mapping_path - path to ICD10 gem file for ICD9-10 mappings

Example run command:

`python 01_generate_datasets.py --train_set_prop 0.5 --mapping_path 2018_I10gem.txt file_path output_dir `
"""
import argparse
import os
import pickle
import random

import pandas as pd


def remove_dot(x):
    return str(x).replace('.', '').strip()


def icd_mapping(row):
    try:
        result = voc[row]
    except KeyError as e:
        end = len(voc) + 1
        voc[e.args[0]] = end
        result = end
    return result


def process_patient(pid, pat_info, processed_dataset, patient_ids, pvc):
    temp_processed_dataset = []
    for visit in pat_info.groupby("DIAGNOSIS_DATE"):
        processed_dataset.append(list(set(list(visit[1]['icd_mapping']))))
        temp_processed_dataset = temp_processed_dataset + list(visit[1]['icd_mapping'])
    pvc.append(list(set(temp_processed_dataset)))
    processed_dataset.append([-1])
    patient_ids.append(pid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--train_set_prop', type=float, default=0.5)
    parser.add_argument('--mapping_path', type=str, default='2018_I10gem.txt')
    args = parser.parse_args()

    # Create output directory if it does not yet exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Map ICD9 to ICD10 to use consistent vocab
    rawdata = pd.read_csv(args.file_path)
    rawdata.columns = ["PATIENT_ID", "PATIENT_BIRTH_DATE", "PATIENT_BIRTH_GENDER", "DIAGNOSIS_CODE",
                       "DIAGNOSIS_DATE", "CODE_SYSTEM", "CASE_FLAG"]
    mapping = pd.read_csv(args.mapping_path, delimiter="\s+", header=None, converters={2: lambda x: str(x)})
    map_dict = dict(zip(mapping[1], mapping[0]))

    rawdata["DIAGNOSIS_CODE"] = rawdata["DIAGNOSIS_CODE"].apply(remove_dot)
    icd9_data = rawdata[rawdata["CODE_SYSTEM"] == "ICD9"]
    icd10_data = rawdata[rawdata["CODE_SYSTEM"] == "ICD10"]

    icd9_data["DIAGNOSIS_CODE"] = icd9_data["DIAGNOSIS_CODE"].map(map_dict)

    mapped_data = pd.concat([icd9_data, icd10_data], ignore_index=True)
    mapped_data = mapped_data.dropna()

    # - Checkpoint: Output mapped raw data
    case_data = mapped_data[mapped_data["CASE_FLAG"] == 1]
    reference_data = mapped_data[mapped_data["CASE_FLAG"] == 0]
    with open(os.path.join(args.output_dir, "reference_dataset_raw.pkl"), 'wb') as ref_raw:
        pickle.dump(reference_data, ref_raw)
    with open(os.path.join(args.output_dir, "case_dataset_raw.pkl"), 'wb') as case_raw:
        pickle.dump(case_data, case_raw)

    # Generate an index to ICD code map dictionary used for both med2vec and later use
    unique_codes = mapped_data.groupby(
        'DIAGNOSIS_CODE'
    ).count()["PATIENT_ID"].reset_index(
        name="count"
    ).sort_values(["count"], ascending=False).reset_index(drop=True)
    keys = list(unique_codes["DIAGNOSIS_CODE"].values)
    values = list(unique_codes.index + 1)
    voc = dict(zip(keys, values))

    print("Vocabulary Size:", len(unique_codes))

    # Now generate Med2Vec-compliant datasets while also doing appropriate splitting into train/reference/case datasets
    mapped_data["icd_mapping"] = mapped_data["DIAGNOSIS_CODE"].apply(icd_mapping)
    groupby_result = mapped_data.groupby(["PATIENT_ID", "CASE_FLAG"])
    reference_processed_dataset = []
    reference_patient_ids = []
    reference_pvc = []
    train_processed_dataset = []
    train_patient_ids = []
    train_pvc = []
    case_processed_dataset = []
    case_patient_ids = []
    case_pvc = []

    for patient_i in groupby_result:
        patient_is_case = patient_i[0][1] == 1
        patient_df = patient_i[1]
        if patient_is_case:
            process_patient(patient_i[0][0], patient_df, case_processed_dataset, case_patient_ids, case_pvc)
        else:
            if random.random() < args.train_set_prop:
                process_patient(patient_i[0][0], patient_df, train_processed_dataset, train_patient_ids, train_pvc)
            else:
                process_patient(patient_i[0][0], patient_df, reference_processed_dataset, reference_patient_ids, reference_pvc)

    # Write Med2Vec Vectors
    with open(os.path.join(args.output_dir, "train_med2vec_vectors.pkl"), 'wb') as f:
        pickle.dump(train_processed_dataset, f)
    with open(os.path.join(args.output_dir, "reference_med2vec_vectors.pkl"), 'wb') as f:
        pickle.dump(reference_processed_dataset, f)
    with open(os.path.join(args.output_dir, "case_med2vec_vectors.pkl"), 'wb') as f:
        pickle.dump(case_processed_dataset, f)

    # Write index to patient ID mappings
    with open(os.path.join(args.output_dir, "train_patient_indexes.pkl"), 'wb') as f:
        pickle.dump(train_patient_ids, f)
    with open(os.path.join(args.output_dir, "reference_patient_indexes.pkl"), 'wb') as f:
        pickle.dump(reference_patient_ids, f)
    with open(os.path.join(args.output_dir, "case_patient_indexes.pkl"), 'wb') as f:
        pickle.dump(case_patient_ids, f)

    # Write index to patient vector mappings
    with open(os.path.join(args.output_dir, "train_patient_vectors.pkl"), 'wb') as f:
        pickle.dump(train_pvc, f)
    with open(os.path.join(args.output_dir, "reference_patient_vectors.pkl"), 'wb') as f:
        pickle.dump(reference_pvc, f)
    with open(os.path.join(args.output_dir, "case_patient_vectors.pkl"), 'wb') as f:
        pickle.dump(case_pvc, f)

    # Write dictionary
    with open(os.path.join(args.output_dir, "vocab_indexes.pkl"), 'wb') as f:
        pickle.dump(voc, f)
