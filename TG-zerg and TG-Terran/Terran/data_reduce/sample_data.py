import numpy as np
from pysc2.lib import point
from pysc2.lib import transform


def func3(name, type_num, train_ratio):
    orders = np.loadtxt("../data/new_data/" + name + ".txt")
    for i in range(10):
        np.random.shuffle(orders)

    seperate_index = int(len(orders) * train_ratio)
    orders_train = orders[:seperate_index, :]
    orders_val = orders[seperate_index:, :]
    func4(orders_train, name, type_num, 'train')
    func4(orders_val, name, type_num, 'val')


def func4(orders, name, type_num, save_name):
    label = orders[:, -3:]

    # high_goal: 0, 1, 2
    label_index, label_count = [], []
    for i in range(type_num):
        label_index.append(np.where(label[:, i])[0])
        label_count.append(len(label_index[i]))

    sample_num = max(label_count)
    print("sample_num:", sample_num)
    label_index_sample = np.array([])
    for i in range(type_num):
        label_index_sample = np.append(label_index_sample, label_index[i])
        add_data = np.random.choice(label_index[i], sample_num - len(label_index[i]))
        label_index_sample = np.append(label_index_sample, add_data)

    for i in range(10):
        np.random.shuffle(label_index_sample)

    order_sample = orders[label_index_sample.astype('int'), :]
    np.savetxt('../data/new_data/' + name + '_' + save_name + '.txt', order_sample)


def func1(name, type_num):
    orders = np.loadtxt("../data/new_data/" + name + ".txt")
    label = orders[:, -3:]

    # high_goal: 0, 1, 2
    label_index, label_count = [], []
    for i in range(type_num):
        label_index.append(np.where(label[:, i])[0])
        label_count.append(len(label_index[i]))

    sample_num = max(label_count)
    print("sample_num:", sample_num)
    label_index_sample = np.array([])
    for i in range(type_num):
        label_index_sample = np.append(label_index_sample, label_index[i])
        add_data = np.random.choice(label_index[i], sample_num - len(label_index[i]))
        label_index_sample = np.append(label_index_sample, add_data)

    for i in range(10):
        np.random.shuffle(label_index_sample)

    order_sample = orders[label_index_sample.astype('int'), :]
    np.savetxt('../data/new_data/' + name + '_sample.txt', order_sample)


def func2(name, type_num):
    orders = np.loadtxt("../data/new_data/" + name + ".txt")
    label = orders[:, -1]

    label_index, label_count = [], []
    for i in range(type_num):
        label_index.append(np.where(label == i)[0])
        label_count.append(len(label_index[i]))
        print(i, ":", len(label_index[i]))

    sample_num = 20000
    print("sample_num:", sample_num)
    label_index_sample = np.array([])
    for i in range(type_num):
        label_index_sample = np.append(label_index_sample, label_index[i])
        add_data = np.random.choice(label_index[i], sample_num - len(label_index[i]))
        label_index_sample = np.append(label_index_sample, add_data)

    for i in range(10):
        np.random.shuffle(label_index_sample)

    order_sample = orders[label_index_sample.astype('int'), :]
    np.savetxt('../data/new_data/' + name + '_sample.txt', order_sample)


if __name__ == "__main__":
    # func3('high', 3, 0.7)
    # func2('tech', 4)
    func2('pop', 5)
