import numpy as np

def find_optimal_split(lst):
    splits = np.arange(0+0.5,len(lst)-0.5)
    optimal_loss = 1e16
    optimal_split = -1
    for split in splits:
        index = int(np.round(split+0.1))
        lst1 = lst[:index]
        lst2 = lst[index:]
        loss = np.sum(np.square(lst1-np.mean(lst1))) + np.sum(np.square(lst2-np.mean(lst2)))
        if loss < optimal_loss:
            optimal_split = split
            optimal_loss = loss
            residuals = np.concatenate((lst[:index] - np.mean(lst1),lst[index:] - np.mean(lst2)))
            tmp1 = np.repeat(np.mean(lst1),len(lst1))
            tmp2 = np.repeat(np.mean(lst2),len(lst2))
            new_lst = np.concatenate((tmp1,tmp2))


    return optimal_loss,optimal_split,residuals,new_lst

def gbdt(err_threshold,lst):
    cur_loss = 1e16
    splits = set()
    round = 0
    while cur_loss > err_threshold:
        round += 1
        print('Round '+str(round))
        loss,split,r,new_lst = find_optimal_split(lst)
        if round == 1:
            res = new_lst
        else:
            res = res + new_lst
        lst = r
        cur_loss = loss
        splits.add(split)
        print(res)
        print(splits)
        print(cur_loss)
    return res,splits



if __name__ == '__main__':
    y = np.array([5.56,5.70,5.91,6.40,6.80,7.05,8.90,8.70,9.00,9.05])
    res,splits = gbdt(0.2,y)
    # loss1,split1,r1,new_lst1 = find_optimal_split(y) # step1
    # print(loss1)
    # print(split1)
    # print(r1)
    # print(new_lst1)
    # loss2, split2, r2, new_lst2 = find_optimal_split(r1)  # step2
    # print(loss2)
    # print(split2)
    # print(r2)
    # print(new_lst2+new_lst1)