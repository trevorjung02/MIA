import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import auc, roc_curve

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# plot data 
def sweep(score, x):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    fpr, tpr, _ = roc_curve(x, -score)
    acc = np.max(1-(fpr+(1-tpr))/2)
    return fpr, tpr, auc(fpr, tpr), acc


def do_plot(prediction, answers, sweep_fn=sweep, metric='auc', legend="", output_dir=None, plot=False):
    """
    Generate the ROC curves by using ntest models as test models and the rest to train.
    """
    fpr, tpr, auc, acc = sweep_fn(np.array(prediction), np.array(answers, dtype=bool))

    tpr_fpr_scores = []
    fpr_thresholds = [0.001, 0.01]
    for fpr_threshold in fpr_thresholds:
        tpr_fpr_scores.append(tpr[np.where(fpr<fpr_threshold)[0][-1]])
    # bp()
    s = 'Attack %s   AUC %.4f, Accuracy %.4f'%(legend, auc,acc)
    for tpr_score, fpr_threshold in zip(tpr_fpr_scores, fpr_thresholds):
        s += ', TPR@%.3f%%FPR of %.4f'%(fpr_threshold * 100, tpr_score)
    print(s)

    metric_text = ''
    if metric == 'auc':
        metric_text = 'auc=%.3f'%auc
    elif metric == 'acc':
        metric_text = 'acc=%.3f'%acc

    if plot:
        plt.plot(fpr, tpr, label=legend+ " " + metric_text)
        # plt.plot(fpr, tpr, label=legend
    return legend, auc,acc, tpr_fpr_scores


def fig_fpr_tpr(all_output, output_dir):
    answers = []
    metric2predictions = defaultdict(list)
    for ex in all_output:
        # bp()
        answers.append(ex["label"])
        for metric in ex["pred"].keys():
            metric2predictions[metric].append(ex["pred"][metric])

    fpr_thresholds = [0.001, 0.01]
    plt.figure(figsize=(4,3))
    with open(f"{output_dir}/auc.txt", "w") as f:
        for metric, predictions in metric2predictions.items():
            # print(metric)
            # print(predictions)
            legend, auc, acc, tpr_fpr_scores = do_plot(predictions, answers, legend=metric, metric='auc', output_dir=output_dir, plot=True)
            s = 'AUC %.4f, Accuracy %.4f'%(auc, acc)
            for tpr_score, fpr_threshold in zip(tpr_fpr_scores, fpr_thresholds):
                s += ', TPR@%.3f%%FPR of %.4f'%(fpr_threshold * 100, tpr_score)
            s += ' | %s\n'%legend
            f.write(s)

    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e-5,1)
    plt.ylim(1e-5,1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], ls='--', color='gray')
    plt.subplots_adjust(bottom=.18, left=.18, top=.96, right=.96)
    plt.legend(fontsize=8)
    plt.savefig(f"{output_dir}/auc.png")
    # plt.show()

def fig_fpr_tpr_plots(all_output, output_dir, metrics):
    answers = []
    metric2predictions = defaultdict(list)
    for ex in all_output:
        # bp()
        answers.append(ex["label"])
        for metric in ex["pred"].keys():
            if metric in metrics or 'RMIA(' in metric or ("RMIA mean" in metric and "RMIA mean ratios" not in metric):
                metric2predictions[metric].append(ex["pred"][metric])
    
    plt.figure(figsize=(4,3))
    best_rmia = None
    rmia_auc = 0
    rmia_predictions = None
    for metric, predictions in metric2predictions.items():
        # print(metric)
        # print(predictions)
        if 'RMIA(' in metric:
            legend, auc, acc, low = do_plot(predictions, answers, legend=metric, metric='auc', output_dir=output_dir, plot=False)
            if auc > rmia_auc:
                best_rmia = metric
                rmia_auc = auc
                rmia_predictions = predictions
        elif "RMIA mean" in metric:
            do_plot(predictions, answers, legend="RMIA mean", metric='auc', output_dir=output_dir, plot=True)
        else:
            do_plot(predictions, answers, legend=metric, metric='auc', output_dir=output_dir, plot=True)
    print(best_rmia)
    print(rmia_auc)
    # print(rmia_predictions)
    do_plot(rmia_predictions, answers, legend="RMIA", metric='auc', output_dir=output_dir, plot=True)
    
    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e-5,1)
    plt.ylim(1e-5,1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], ls='--', color='gray')
    plt.subplots_adjust(bottom=.18, left=.18, top=.96, right=.96)
    plt.legend(fontsize=8)
    plt.savefig(f"{output_dir}/auc.png")
    # plt.show()