import numpy as np

def find_optimal_splits(labels,weights):
    splits = np.arange(0 - 0.5, len(labels) - 0.5)
    optimal_loss = 1e16
    optimal_split = -1
    optimal_direction = -1 # -1 represents that when i < split , the label is -1, otherwise the label is 1.
    for split in splits:
        index = np.round(split+0.1)
        lst1 = labels[:index]
        lst2 = labels[index:]
        #calculate the losses of both directions
        directions = [-1,1]
        for direction in directions:
            cur_loss = weights[:index] * np.array([1 if lst1[i] != direction else 0 for i in range(len(lst1))])
            cur_loss += weights[index:] * np.array(1 if lst2[i] != - direction else 0 for i in range(len(lst2)))
            if cur_loss < optimal_loss:
                optimal_split = split
                optimal_loss = cur_loss
                optimal_direction = direction
    alpha = 0.5 * np.log(optimal_loss / (1 - optimal_loss))
    return optimal_split,optimal_direction,optimal_loss,alpha

def next_coefficents(labels,weights):
    split,direction,loss = find_optimal_splits(labels,weights)
    alpha = 0.5*np.log(loss/(1-loss))
    predict_labels = [direction if i < split else -direction for i in range(labels)]
    new_weights = np.array([alpha if labels[j] != predict_labels[j] else -alpha for j in range(labels)])
    new_weights = np.power(new_weights,np.e)
    new_weights = weights * new_weights / np.sum(new_weights)
    return new_weights

def adaboost(threshold,labels):
    weights = np.repeat(1./len(labels),len(labels))
    split, direction, loss,alpha = find_optimal_splits()
    splits = [split]
    alphas = [alpha]
    while loss > threshold:
        new_weights = next_coefficents(labels,weights)
        split, direction, loss = find_optimal_splits()
        splits.append(split)
        alphas.append(alpha)







