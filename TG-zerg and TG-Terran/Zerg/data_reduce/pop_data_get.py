import numpy as np


def reduce():
    orders = np.loadtxt("../data/new_data/pop.txt")
    label = orders[:, -1]

    label_index = []
    label_index.extend(np.where(label == 0)[0])
    label_index.extend(np.where(label == 3)[0])
    label_index.extend(np.where(label == 4)[0])

    new_orders = orders[label_index]

    for i in range(new_orders.shape[0]):
        if new_orders[i, -1] == 3:
            new_orders[i, -1] = 1
        elif new_orders[i, -1] == 4:
            new_orders[i, -1] = 2

    np.savetxt("../data/new_data/new_pop.txt", new_orders)


def sample(name, type_num):
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
    reduce()
    sample("new_pop", 3)