import numpy as np
import os


def gather_data():
    path = "../data/new_data/"
    dirs = os.listdir(path)
    num = len(dirs)
    print("Find %d dirs for gather" % num)

    order_data = np.array([])
    for data_dir in dirs:
        order = np.loadtxt(path + data_dir + "/" + "order.txt")
        num = order.shape[0]
        temp = np.zeros(order.shape[1] + 1)
        temp[0] = int(data_dir)

        for index in range(num):
            if order[index, 1] != 4 and order[index, [2, 3]].sum() != 0:
                temp[1:] = order[index]
                order_data = np.append(order_data, temp)

    np.savetxt("../data/new_order.txt", order_data.reshape(-1, order.shape[1] + 1))


def sample_data():

    orders = np.loadtxt("../data/new_order.txt")
    label = orders[:, 2]

    # action_type: 0 : move, 1 : build_pylon, 2 : build_forge, 3: build_cannon
    type_num = 4
    label_index = []
    for i in range(type_num):
        label_index.append(np.where(label == i)[0])

    # each label sample some times
    sample_num = 1000
    label_index_sample = np.array([])
    for i in range(type_num):
        info = np.random.choice(label_index[i], sample_num)
        label_index_sample = np.append(label_index_sample, info)

    # shuffle ten times
    for i in range(10):
        np.random.shuffle(label_index_sample)

    order_sample = orders[label_index_sample.astype('int'), :]
    np.savetxt('../data/new_order_sample.txt', order_sample)


if __name__ == "__main__":
    # gather_data()
    sample_data()
