"""
Parse WordNet and FB15k datasets
"""

import os
import json
import pickle
import argparse
import ipdb
from dataloading import load_data

if 'data' not in os.listdir('./'):
    os.mkdir('./data')


def get_idx_dicts(data):
    ent_set, rel_set = set(), set()
    for lhs, rel, rhs in data:
        ent_set.add(lhs)
        rel_set.add(rel)
        ent_set.add(rhs)
    ent_list = sorted(list(ent_set))
    rel_list = sorted(list(rel_set))

    ent_to_idx, rel_to_idx = {}, {}
    for i, ent in enumerate(ent_list):
        ent_to_idx[ent] = i
    for j, rel in enumerate(rel_list):
        rel_to_idx[rel] = j
    return ent_to_idx, rel_to_idx

def transform_data(data, ent_to_idx, rel_to_idx):
    dataset = []
    for lhs, rel, rhs in data:
        dataset += [[ent_to_idx[lhs], rel_to_idx[rel], ent_to_idx[rhs]]]
    return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="Choose dataset for parsing")
    args = parser.parse_args()
    if args.dataset in ['facebook','pokec_z','pokec_n','nba','twitter','german',
                        'recidivism','credit','google+','LCC','LCC_small']:
        adj, features, labels, edges, idx_train, idx_val, idx_test, sens, sens_idx = load_data(args.dataset)
    else:
        raise Exception("Argument 'dataset' did not have a valid input.")
    
    # coo_adj = adj.coalesce()
    # indices = coo_adj.indices().numpy()
    # values = coo_adj.values().numpy()

    # full_data = [[int(indices[0, i]), int(values[i]), int(indices[1, i])] for i in range(coo_adj._nnz())]

    # idx_train_np = idx_train.numpy()
    # idx_val_np = idx_val.numpy()
    # idx_test_np = idx_test.numpy()

    # train_data = [entry for entry in full_data if entry[0] in idx_train_np]
    # valid_data = [entry for entry in full_data if entry[0] in idx_val_np]
    # test_data = [entry for entry in full_data if entry[0] in idx_test_np]

    all_data = edges.tolist()
    train_e = [i for i in all_data if i[0] in idx_train]
    valid_e = [i for i in all_data if i[0] in idx_val]
    test_e = [i for i in all_data if i[0] in idx_test]

    # existance of relation
    relation = 1 

    train_data = [[edge[0], relation, edge[1]] for edge in train_e]
    test_data = [[edge[0], relation, edge[1]] for edge in test_e]
    valid_data = [[edge[0], relation, edge[1]] for edge in valid_e]
    ipdb.set_trace()

    ent_to_idx, rel_to_idx = get_idx_dicts(train_data + valid_data + test_data)

    train_set = transform_data(train_data, ent_to_idx, rel_to_idx)
    valid_set = transform_data(valid_data, ent_to_idx, rel_to_idx)
    test_set = transform_data(test_data, ent_to_idx, rel_to_idx)

    pickle.dump(train_set, open('./data/%s-train.pkl' % args.dataset, 'wb'), protocol=-1)
    pickle.dump(valid_set, open('./data/%s-valid.pkl' % args.dataset, 'wb'), protocol=-1)
    pickle.dump(test_set, open('./data/%s-test.pkl' % args.dataset, 'wb'), protocol=-1)

    json.dump(ent_to_idx, open('./data/%s-ent_to_idx.json' % args.dataset, 'w'))
    json.dump(rel_to_idx, open('./data/%s-rel_to_idx.json' % args.dataset, 'w'))

    print("Dataset: %s" % args.dataset)
    print("# entities: %s; # relations: %s" % (len(ent_to_idx),
                                               len(rel_to_idx)))
    print("train set size: %s; valid set size: %s; test set size: %s" % (len(train_set),
                                                                         len(valid_set),
                                                                         len(test_set)))

if __name__ == '__main__':
    main()
