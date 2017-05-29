from pprint import pprint
from random import randint, random, seed
from math import exp, copysign, sqrt

seed(1)

def generate_learn_set():
    learn_set = []
    for i in range(500):
        x, y, z = randint(-50, 50), randint(-50, 50), randint(-50, 50)

        if x >= 0 and y >= 0 and z >= 0:
            quadrant = (1, 0, 0, 0, 0, 0, 0, 0,)
        if x >= 0 and y < 0 and z >= 0:
            quadrant = (0, 1, 0, 0, 0, 0, 0, 0,)
        if x < 0 and y < 0 and z >= 0:
            quadrant = (0, 0, 1, 0, 0, 0, 0, 0,)
        if x < 0 and y >= 0 and z >= 0:
            quadrant = (0, 0, 0, 1, 0, 0, 0, 0,)
        if x >= 0 and y >= 0 and z < 0:
            quadrant = (0, 0, 0, 0, 1, 0, 0, 0,)
        if x >= 0 and y < 0 and z < 0:
            quadrant = (0, 0, 0, 0, 0, 1, 0, 0,)
        if x < 0 and y < 0 and z < 0:
            quadrant = (0, 0, 0, 0, 0, 0, 1, 0,)
        if x < 0 and y >= 0 and z < 0:
            quadrant = (0, 0, 0, 0, 0, 0, 0, 1,)
        learn_set.append(((x, y, z), quadrant))
    pprint(learn_set)
    return learn_set


def get_neurons():
    neurons = []
    for i in range(8):
        neurons.append([])

    for i, n in enumerate(neurons):
        for j in range(3):
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
            print(i / 10)
        for (x, y, z), q in learn_set:
            # run the net, gather outputs
            res = []
            for n in neurons:
                vlen = sqrt(x*x+y*y+z*z)
                NET = x/vlen * n[0] + y/vlen * n[1] + z/vlen * n[2]
                if abs(NET) > 700:
                    NET = copysign(700, NET)
                OUT = sigmoid(NET)
                res.append(OUT)

            # calc error
            errors = []
            for j, output in enumerate(q):
                error = output - res[j]
                errors.append(error)

            # write biggest error
            # recalc weights
            for j, n in enumerate(neurons):
                n[0] += sigmoid_derivative(res[j]) * x * errors[j]
                n[1] += sigmoid_derivative(res[j]) * y * errors[j]
                n[2] += sigmoid_derivative(res[j]) * z * errors[j]
    pprint(neurons)
    return neurons



def use(neurons, input_vec):
    x, y, z = input_vec
    res = []
    for n in neurons:
        vlen = sqrt(x*x+y*y+z*z)
        NET = x/vlen * n[0] + y/vlen * n[1] + z/vlen * n[2]
        if abs(NET) > 700:
            NET = copysign(700, NET)
        OUT = sigmoid(NET)
        res.append(int(OUT))
    return res


if __name__ == '__main__':
    neurons = learn(generate_learn_set())
    pprint(use(neurons, (-33, 22, 47)))
