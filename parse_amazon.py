"""
Parse WordNet and FB15k datasets
"""

import os
import json
import pickle
import argparse
import ipdb

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
    parser.add_argument('--dataset', help="Choose to parse Amazon-2")
    args = parser.parse_args()
    if args.dataset == 'Amazon-2':
        path = "../Graph-Mining-Fairness-Data/dataset/" + args.dataset + "/%s_df.pkl"
    else:
        raise Exception("Argument 'dataset' can only be Amazon-2.")

    train_file = pickle.load(open(path % 'training', 'rb'))
    valid_file = pickle.load(open(path % 'valiing', 'rb'))
    test_file = pickle.load(open(path % 'testing', 'rb'))
    columns = train_file.columns.tolist()

    
    train_data = train_file[[columns[0], columns[2], columns[1]]].values.tolist()
    valid_data = valid_file[[columns[0], columns[2], columns[1]]].values.tolist()
    test_data = test_file[[columns[0], columns[2], columns[1]]].values.tolist()

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
