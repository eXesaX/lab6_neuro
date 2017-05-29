from pprint import pprint
from random import randint, random, seed
from math import exp, copysign, sqrt
from math import sin, cos, radians
from numpy import linspace
from numpy.random import normal
seed(1)


def get_cluster(num_of_points, d, X, Y):
    points = []
    norm = normal(0, d, num_of_points)
    for i, k in enumerate(linspace(0, 360, num_of_points)):
        x = X + norm[i] * cos(radians(k))
        y = Y + norm[i] * sin(radians(k))
        points.append((x.item(), y.item()))
    return points


def generate_learn_set():
    learn_set = []
    first = get_cluster(30, 30, -50, -50)
    second = get_cluster(30, 30, 50, 50)
    for x, y in first:
        learn_set.append((x, y, 0))
    for x, y in second:
        learn_set.append((x, y, 1))
    pprint(learn_set)
    return learn_set


def get_neurons():
    neurons = []
    for i in range(2):
        neurons.append([])

    for i, n in enumerate(neurons):
        for j in range(2):
            neurons[i].append(random() * 2 - 1)
    pprint(neurons)
    return neurons


def sigmoid(x):
    try:
        return 1 / (1 + 2.71 ** (-x))
    except OverflowError:
        print(x)


def sigmoid_derivative(x):
    return x * (1 - x)


def learn(learn_set):
    neurons = get_neurons()
    for i in range(1000):
        if i % 100 == 0:
            print(i / 10)
        for x, y, t in learn_set:
            # run the net, gather outputs
            res = []
            maxnet = None
            maxnet_j = 0
            for j, n in enumerate(neurons):
                vlen = sqrt(x*x+y*y)
                NET = x/vlen * n[0] + y/vlen * n[1]
                OUT=NET
                if maxnet is None:
                    maxnet = OUT
                    maxnet_j = j
                if OUT > maxnet:
                    maxnet = OUT
                    maxnet_j = j
                res.append(OUT)

            for j, r in enumerate(res):
                res[j] = 0
            res[maxnet_j] = 1

            # calc error
            # errors = []
            # for j, output in enumerate(res):
            #     error = t - res[j]
            #     errors.append(error)

            # write biggest error
            # recalc weights
            # for j, n in enumerate(neurons):
            alpha = 0.0001 # (50 - i) / 100
            neurons[maxnet_j][0] = neurons[maxnet_j][0] + alpha * (x - neurons[maxnet_j][0])# sigmoid_derivative(res[j]) * x * errors[j]
            neurons[maxnet_j][1] = neurons[maxnet_j][1] + alpha * (y - neurons[maxnet_j][1])#sigmoid_derivative(res[j]) * x * errors[j]
            # n[1] += sigmoid_derivative(res[j]) * y * errors[j]
    pprint(neurons)
    return neurons



def use(neurons, input_vec):
    x, y = input_vec
    res = []
    maxnet = None
    maxnet_j = 0
    for j, n in enumerate(neurons):
        vlen = sqrt(x*x+y*y)
        NET = x/vlen * n[0] + y/vlen * n[1]
        OUT=NET
        if maxnet is None:
            maxnet = OUT
            maxnet_j = j
        if OUT > maxnet:
            maxnet = OUT
            maxnet_j = j
        res.append(OUT)
    for j, r in enumerate(res):
        res[j] = 0
    res[maxnet_j] = 1
    return res


if __name__ == '__main__':
    neurons = learn(generate_learn_set())
    pprint(use(neurons, (50, 50)))
