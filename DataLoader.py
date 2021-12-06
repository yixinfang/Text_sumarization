import collections
import os

LOAD_PATH = './DUC2004 Dataset/DUC2004_Summarization_Documents/duc2004_testdata/tasks1and2/duc2004_tasks1and2_docs/docs'
REFERENCE_PATH = './DUC2004 Dataset/reference'


# Helper function to load all documents for DUC task1 and task2 into a dictionary
def task1and2Loader():
    d = collections.defaultdict(dict)
    for folder_num in os.listdir(LOAD_PATH):
        for doc in os.listdir(os.path.join(LOAD_PATH, folder_num)):
            with open(os.path.join(LOAD_PATH, folder_num, doc), "r", encoding="utf8") as f:
                sentences = f.read().splitlines()
                full_doc = ' '.join([i.strip() for i in sentences[1:]])
                doc_name = doc.split(".")[0]
                d[f'Task{folder_num}'][doc_name] = full_doc
    return d


# Helper function to load all references for task1 and task2 into a dictionary
def task1and2ReferenceLoader():
    d = collections.defaultdict(dict)
    for ref in os.listdir(REFERENCE_PATH):
        filename = ref.split(".")[0]
        task_name, ref_name = filename.split("_")
        if ref_name not in ['reference1', 'reference2']:
            continue
        with open(os.path.join(REFERENCE_PATH, ref), "r", encoding="utf8") as f:
            sentences = f.read().splitlines()
            full_doc = ' '.join([i.strip() for i in sentences])
            d[ref_name][task_name] = full_doc
    return d
