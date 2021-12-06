import collections
import os
import torch
import DataLoader
import pickle
from rouge import Rouge
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer, pipeline


# A wrapper class for Bart model summarization
class BartModel:
    def __init__(self):
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    def summarize(self, document, min_count=None, max_count=None):
        inputs = self.tokenizer.batch_encode_plus([document], return_tensors='pt', Truncation=True)
        summary_ids = self.model.generate(inputs['input_ids'], num_beams=4, early_stopping=True,
                                          max_length=max_count, min_length=min_count)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    # summarize for document length that are greater than 1024 tokens (maximum for BART tokenizer)
    # following method credits to: https://discuss.huggingface.co/t/summarization-on-long-documents/920/23?page=2
    def summarize_for_long_text(self, long_text, max_length=40):
        # tokenize without truncation
        inputs_no_trunc = self.tokenizer(long_text, max_length=None, return_tensors='pt', truncation=False)

        # get batches of tokens corresponding to the exact model_max_length
        chunk_start = 0
        chunk_end = self.tokenizer.model_max_length  # == 1024 for Bart
        inputs_batch_lst = []
        while chunk_start <= len(inputs_no_trunc['input_ids'][0]):
            inputs_batch = inputs_no_trunc['input_ids'][0][chunk_start:chunk_end]  # get batch of n tokens
            inputs_batch = torch.unsqueeze(inputs_batch, 0)
            inputs_batch_lst.append(inputs_batch)
            chunk_start += self.tokenizer.model_max_length  # == 1024 for Bart
            chunk_end += self.tokenizer.model_max_length  # == 1024 for Bart

        # generate a summary on each batch
        summary_ids_lst = [self.model.generate(inputs, num_beams=4, max_length=max_length, early_stopping=True) for inputs in
                           inputs_batch_lst]

        # decode the output and join into one string with one paragraph per summary batch
        summary_batch_lst = []
        for summary_id in summary_ids_lst:
            summary_batch = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for
                             g in
                             summary_id]
            summary_batch_lst.append(summary_batch[0])
        summary_all = ' '.join(summary_batch_lst)
        return summary_all


# write summaries as text file.
def write_summary(dict, task_name):
    folder_path = "Generated Summaries using BART"
    file_path = os.path.join(folder_path, task_name)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    for t in dict:
        name_t = t.replace("Task", "Doc")
        with open(os.path.join(file_path, f"{name_t}_summary.txt"), "w") as file:
            summary = dict[t]
            summary = summary.replace(". ", ".\n ")
            file.write(summary)


# save dictionary as pickle file
def save_pickle(dict, name):
    with open(f'{name}.pickle', 'wb') as handle:
        pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


# load pickle file
def load_pickle(name):
    with open(f'{name}.pickle', 'rb') as handle:
        return pickle.load(handle)


# helper function to calculate the rouge score
def calculate_rouge(sum_dict, ref_dict, ref_name):
    rouge = Rouge()
    cur_ref = ref_dict[ref_name]
    summary_ordered_list = [i[1] for i in list(sorted(sum_dict.items(), key=lambda item: item[0]))]
    reference_ordered_list = [i[1] for i in list(sorted(cur_ref.items(), key=lambda item: item[0]))]
    rouge_dict = rouge.get_scores(summary_ordered_list, reference_ordered_list, avg=True)
    return rouge_dict


if __name__ == "__main__":

    print("Hey")
    dataset = DataLoader.task1and2Loader()
    bart = BartModel()

    # # Task 1 (Commented out the generating summaries part)
    # # For generating a dictionary of summaries for each document in each cluster, and save it in a pickle.
    # task1_dict = collections.defaultdict(dict)
    # for task in tqdm(dataset):
    #     for doc in tqdm(dataset[task]):
    #         task1_dict[task][doc] = bart.summarize(dataset[task][doc])
    # save_pickle(task1_dict, "generated_summaries")

    # task1_dict = load_pickle("generated_summaries")
    # task1_summaries = {}
    # for task in tqdm(task1_dict):
    #     task1_dict[task] = " ".join(list(task1_dict[task].values()))
    #     task1_summaries[task] = bart.summarize(task1_dict[task], 100, 200)
    # save_pickle(task1_summaries, "task1_summaries_100_120")

    # # Task 2
    # task2_summaries = {}
    # for task in tqdm(dataset):
    #     doc_cluster = " ".join(list(dataset[task].values()))
    #     task2_summaries[task] = bart.summarize_for_long_text(doc_cluster, max_length=25)
    # save_pickle(task2_summaries, "task2_summaries_25")

    task1_summaries = load_pickle("task1_summaries_100_120")
    task2_summaries = load_pickle("task2_summaries_150_250")
    reference = DataLoader.task1and2ReferenceLoader()
    # write_summary(task1_summaries, "Task1")
    # write_summary(task2_summaries, "Task2")

    rouge_task1 = calculate_rouge(task1_summaries, reference, 'reference1')
    rouge_task2 = calculate_rouge(task2_summaries, reference, 'reference2')
    print(rouge_task1)
    print(rouge_task2)
