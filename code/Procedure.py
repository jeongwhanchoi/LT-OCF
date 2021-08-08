'''
Design training and test process
'''
import world
import numpy as np
import torch
import utils
from utils import timer
import model
import multiprocessing
from time import perf_counter

CORES = multiprocessing.cpu_count() // 2

def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.

    t = perf_counter()
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri         
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    train_time = perf_counter()-t
    print("Train time: {:.4f}s".format(train_time))
    return f"loss{aver_loss:.3f}-{time_info}"
    

def BPR_train_ode_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLossT = loss_class
    
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.

    t = perf_counter()
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri, all_time = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        with torch.no_grad():
            for j in range(len(all_time)):
                if j == 0:
                    start = 0 + 0.01
                else:
                    start = all_time[j - 1].item() + 0.01

                if j == len(all_time) - 1:
                    end = 1 - 0.01
                else:
                    end = all_time[j + 1].item() - 0.01
                all_time[j] = all_time[j].clamp_(min=start, max=end)           
        times =  {'t'+str(t+1):time.item() for t, time in enumerate(all_time)}
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
            w.add_scalars(f'time_points', times, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    train_time = perf_counter()-t
    print("Train time: {:.4f}s".format(train_time))
    return f"loss{aver_loss:.3f}-{time_info}", times
    
def BPR_train_ode(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLossT = loss_class
    
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.

    t = perf_counter()
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri, all_times = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        ceiling = 0 #TODO: convert to an argument
        with torch.no_grad():
            if ceiling == 0:
                constraint_value = 0.000001
                final_time = world.config['K']
                all_times[0] = all_times[0].clamp_(min=0+constraint_value, max=all_times[1].item()-constraint_value)
                all_times[0] = all_times[0].clamp_(min=0+constraint_value, max=final_time- 3*constraint_value)
                all_times[1] = all_times[1].clamp_(min=all_times[0].item()+constraint_value, max=all_times[2].item()-constraint_value)
                all_times[1] = all_times[1].clamp_(min=all_times[0].item()+constraint_value, max=final_time-2*constraint_value)
                all_times[2] = all_times[2].clamp_(min=all_times[1].item()+constraint_value, max=final_time-constraint_value)
            elif ceiling == 1:
                for n, time_n in enumerate(all_times):
                    start = n
                    end = n + 1
                    time_n[0] = time_n[0].clamp_(min=start, max=end)

        times_0 =  {'t'+str(t+1):time.item() for t, time in enumerate(all_times[0])}
        times_1 =  {'t'+str(t+1):time.item() for t, time in enumerate(all_times[1])}
        times_2 =  {'t'+str(t+1):time.item() for t, time in enumerate(all_times[2])}
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
            w.add_scalars(f'time_points/time_points_1', times_0, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
            w.add_scalars(f'time_points/time_points_2', times_1, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
            w.add_scalars(f'time_points/time_points_3', times_2, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    train_time = perf_counter()-t
    print("Train time: {:.4f}s".format(train_time))
    return f"loss{aver_loss:.3f}-{time_info}", [times_0, times_1, times_2]

def BPR_train_ode_t2(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLossT = loss_class
    
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri, all_times = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        ceiling = 0 #TODO: convert to an argument
        with torch.no_grad():
            if ceiling == 0:
                constraint_value = 0.000001
                final_time = world.config['K']
                all_times[0] = all_times[0].clamp_(min=0+constraint_value, max=all_times[1].item()-constraint_value)
                all_times[0] = all_times[0].clamp_(min=0+constraint_value, max=final_time- 2*constraint_value)
                all_times[1] = all_times[1].clamp_(min=all_times[0].item()+constraint_value, max=final_time-constraint_value)
            elif ceiling == 1:
                for n, time_n in enumerate(all_times):
                    start = n
                    end = n + 1
                    time_n[0] = time_n[0].clamp_(min=start, max=end)

        times_0 =  {'t'+str(t+1):time.item() for t, time in enumerate(all_times[0])}
        times_1 =  {'t'+str(t+1):time.item() for t, time in enumerate(all_times[1])}
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
            w.add_scalars(f'time_points/time_points_1', times_0, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
            w.add_scalars(f'time_points/time_points_2', times_1, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}", [times_0, times_1]

def BPR_train_ode_t1(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLossT = loss_class
    
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri, all_times = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        ceiling = 0 #TODO: convert to an argument
        with torch.no_grad():
            if ceiling == 0:
                constraint_value = 0.000001
                final_time = world.config['K']
                all_times[0] = all_times[0].clamp_(min=0+constraint_value, max=final_time- constraint_value)
            elif ceiling == 1:
                for n, time_n in enumerate(all_times):
                    start = n
                    end = n + 1
                    time_n[0] = time_n[0].clamp_(min=start, max=end)

        times_0 =  {'t'+str(t+1):time.item() for t, time in enumerate(all_times[0])}
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
            w.add_scalars(f'time_points/time_points_1', times_0, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}", times_0
          
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
        
            
def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    if world.model_name == 'ltgccf':
        Recmodel: model.LTGCCF
    else:
        Recmodel: model.LightGCN

    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        t = perf_counter()
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [ 
            #         utils.AUC(rating[i],
            #                   dataset, 
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        inference_time = perf_counter()-t
        print("Inference time: {:.4f}s".format(inference_time))
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
        if multicore == 1:
            pool.close()
        print(results)
        return results
