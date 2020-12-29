from utils import *


def generate_initial(generation_size, num_features):
    generation = []
    neurons_copy = [num_features] + neurons
    num_w = 0
    for i in range(1, len(neurons_copy)):
        num_w += neurons_copy[i-1] * neurons_copy[i]

    for _ in range(generation_size):
        generation.append(np.random.uniform(low=-LIMIT, high=LIMIT, size=(num_w,)))

    return np.array(generation)


def criterion(y_pred, y_true):
    return 1 - (y_pred == y_true).mean(axis=0, keepdims=True).T


def limit(swarm):
    np.clip(swarm, -LIMIT, LIMIT, out=swarm)


def get_best(swarm, f):
    best = f.min()
    arg = np.argmin(f)
    return best, swarm[arg]


def get_vn(SWARM_SIZE, gbest, p_best, vn_1, xn_1):
    w = 0.8
    c1, c2 = 1.494, 1.494
    vmax = 0.2

    rand1 = np.random.rand(SWARM_SIZE, 1)
    rand2 = np.random.rand(SWARM_SIZE, 1)

    vn = w * vn_1 + c1 * rand1 * (p_best-xn_1) + c2 * rand2 * (gbest - xn_1)
    np.clip(vn, -vmax, vmax, out=vn)

    return vn


def pso(x, swarm, y_true, threshold, MAX_GEN):
    SWARM_SIZE = len(swarm)
    deltaT = 1.0

    # Globally best solution
    gbest = float('inf')
    gbest_coord = None
    gbest_vec = np.zeros((1, MAX_GEN))

    # Personal best of each member of the swarm
    y_pred = forward(x, swarm, threshold)
    pbest = criterion(y_pred, y_true)
    pbest_coord = swarm.copy()
    pbest_vec = np.zeros((SWARM_SIZE, MAX_GEN))

    v_n = np.random.uniform(low=-LIMIT, high=LIMIT, size=(SWARM_SIZE, 1))

    for gen in range(MAX_GEN):
        y_pred = forward(x, swarm, threshold)
        f = criterion(y_pred, y_true)

        gbest_curr, best_curr = get_best(swarm, f)

        if gbest_curr < gbest:
            gbest = gbest_curr
            gbest_coord = best_curr

        for i, sol in enumerate(f):
            if sol < pbest[i, 0]:
                pbest[i, 0] = sol
                pbest_coord[i] = swarm[i]

        v_n = get_vn(SWARM_SIZE, gbest_coord, pbest_coord, v_n, swarm)
        swarm = swarm + v_n * deltaT
        limit(swarm)

        gbest_vec[0, gen] = gbest
        pbest_vec[:, gen] = pbest[:, 0].copy()

    y_pred = forward(x, swarm, threshold)
    f = criterion(y_pred, y_true)
    gbest_curr, best_curr = get_best(swarm, f)

    if gbest_curr > gbest:
        gbest = gbest_curr
        gbest_coord = best_curr

    return gbest, gbest_coord, gbest_vec, pbest_vec
