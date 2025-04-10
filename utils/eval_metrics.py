import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """

    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def weighted_accuracy(test_preds_emo, test_truth_emo):
    true_label = (test_truth_emo > 0)
    predicted_label = (test_preds_emo > 0)
    tp = float(np.sum((true_label ==1) & (predicted_label == 1)))
    tn = float(np.sum((true_label ==1) & (predicted_label == 0)))
    p = float(np.sum(true_label ==1))
    n = float(np.sum(true_label ==1))

    return (tp*(n/p) + tn) / (2*n)


def eval_senti(results, truths, exclude_zero = False):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)])

    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

    mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
    f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)
    acc = accuracy_score(binary_truth, binary_preds)

    print("MAE: ", mae)
    print("Correlation coefficient: ", corr)
    print("mult_acc_7: ", mult_a7)
    print("mult_acc_5: ", mult_a5)
    print("F1 score: ", f_score)
    print("Accuracy: ", acc)

    return acc


def eval_food(results, truths):
    test_preds = results.view(-1, 101).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    test_preds_i = np.argmax(test_preds, axis=1)
    test_truth_i = test_truth
    f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
    acc = accuracy_score(test_truth_i, test_preds_i)
    print("    - F1 Score: ", f1)
    print("    - Accuracy: ", acc)
    return acc


def cosine_schedular(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmpup_value = 0.):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmpup_value, base_value, warmup_iters)

    iters = np.arange(epochs*niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value-final_value) * (1+np.cos(np.pi* iters/len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs*niter_per_ep
    return schedule