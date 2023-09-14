import numpy as np
import statistics

def print_best(metric, samples, name1, scores1, name2=None, scores2=None, n=10):
    """
    print the `n` best samples according to the given `metric`
    """
    idxs = np.argsort(metric)[::-1][:n]
    print("idxs: ", idxs)
    acc_list = []
    for id in idxs:
        if id < 10:
            acc_list.append(1)
        else:
            acc_list.append(0)
    print("acc: ", statistics.mean(acc_list))

# print and write mean target model perplexity on seen vs unseen data
def compute_ppl(all_output, output_dir):
    seen_data_avg = []
    unseen_data_avg = []
    for ex in all_output:
        if ex["label"] == 0:
            unseen_data_avg.append(ex["pred"]["ppl"])
        elif ex["label"] == 1:
            seen_data_avg.append(ex["pred"]["ppl"])
    print("seen_data_avg: ", statistics.mean(seen_data_avg))
    print("unseen_data_avg: ", statistics.mean(unseen_data_avg))
    with open(f"{output_dir}/ppl.txt","w") as f:
        f.write(f"seen_data_avg: {statistics.mean(seen_data_avg)}\n")
        f.write(f"unseen_data_avg: {statistics.mean(unseen_data_avg)}\n")

# print and write mean target model top_k perplexity on seen vs unseen data
def compute_topkmean(all_output, output_dir):
    seen_data_avg = []
    unseen_data_avg = []
    for ex in all_output:
        if ex["label"] == 0:
            unseen_data_avg.append(ex["pred"]["topk_mean"])
        elif ex["label"] == 1:
            seen_data_avg.append(ex["pred"]["topk_mean"])
    print("topkmean:seen_data_avg: ", statistics.mean(seen_data_avg))
    print("topkmean:unseen_data_avg: ", statistics.mean(unseen_data_avg))
    with open(f"{output_dir}/var.txt","w") as f:
        f.write(f"topkmean:seen_data_avg: {statistics.mean(seen_data_avg)}\n")
        f.write(f"topkmean:unseen_data_avg: {statistics.mean(unseen_data_avg)}\n")