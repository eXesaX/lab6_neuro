from pprint import pprint
from random import randint, random, seed
from math import exp, copysign
from copy import copy
seed(1)

def generate_learn_set():
    learn_set = []
    for i in range(500):
        x = randint(-50, 50)
        if x >= 0:
            quadrant = 1
        if x < 0:
            quadrant = 0
        learn_set.append((x, quadrant))
    pprint(learn_set)
    return learn_set


def get_neurons():
    neurons = []
    for i in range(1):
        neurons.append([])

    for i, n in enumerate(neurons):
        for j in range(1):
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
    for i in range(10000):
        if i % 100 == 0:
            print(i / 100)
        for x, q in learn_set:
            # run the net, gather outputs
            res = []
            for n in neurons:
                NET = x * n[0]
                if abs(NET) > 700:
                    NET = copysign(700, NET)
                OUT = sigmoid(NET)
                res.append(OUT)

            # calc error
            errors = []
            # for i, output in enumerate(q):
            error = q - res[0]
            errors.append(error)

            # recalc weights
            for j, n in enumerate(neurons):
                try:
                    n[0] = n[0] + sigmoid_derivative(res[j]) * x * errors[j]
                except IndexError:
                    print(errors, j)
    return neurons



def use(neurons, input_vec):
    x = input_vec
    res = []
    for n in neurons:
        NET = x * n[0]
        if abs(NET) > 700:
            NET = copysign(700, NET)
        OUT = sigmoid(NET)
        res.append(int(OUT))
    return res


def guess(neurons, nums):
    print(use(neurons[0], nums[0]), use(neurons[1], nums[1]), use(neurons[2], nums[2]))

if __name__ == '__main__':
    learn_set = generate_learn_set()
    neuron1 = learn(learn_set)
    neuron2 = copy(neuron1)
    neuron3 = copy(neuron1)
    pprint(guess((neuron1, neuron2, neuron3), (20, 20, 20)))
