import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from particle_swarm import *


NUM_RUNS = 20
SWARM_SIZE = 100
MAX_GEN = 50

if __name__ == "__main__":
    features, labels = load_dataset('data_banknote_authentication.txt')
    train_x, test_x, train_y, test_y = train_test_split(features, labels,
                                                        test_size=0.2,
                                                        stratify=labels)
    # The treshold used for the final prediction
    threshold = 1 - train_y.mean()

    acum_train = []
    acum_test = []
    acum_wbest = []

    acum_gbest_vec = np.zeros((NUM_RUNS, MAX_GEN))
    acum_pbest_vec = []

    t1 = time.time()
    for i in range(NUM_RUNS):
        print("Run:", i)
        swarm = generate_initial(SWARM_SIZE, train_x.shape[1])
        gbest, wbest, gbest_vec, pbest_vec = pso(train_x, swarm, train_y, threshold, MAX_GEN)
        acum_train.append(gbest*100)
        acum_wbest.append(wbest)
        acum_gbest_vec[i] = gbest_vec*100
        acum_pbest_vec.append(pbest_vec)

        y_pred = forward(test_x, wbest, threshold)
        acum_test.append(criterion(y_pred, test_y)[0, 0]*100)

    print(f"Execution time: {time.time()-t1:.1f} seconds")
    print(f"Mean train error: {np.mean(acum_train):.5f}%  |  Standard deviation: {np.std(acum_train):.5f}")
    print(f"Mean test error:  {np.mean(acum_test):.5f}%   |  Standard deviation: {np.std(acum_test):.5f}")

    plt.figure(figsize=(8, 5))
    for i in range(NUM_RUNS):
        plt.plot(range(1, MAX_GEN+1), acum_gbest_vec[i, :], label=f"Run: {i+1}")

    plt.title('Globally best error')
    plt.xlabel('Generation')
    plt.ylabel('Error[%]')
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, NUM_RUNS+1), acum_train)
    plt.title('Globally best error: Independent runs')
    plt.xlabel('Run')
    plt.ylabel('Error [%]')
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, MAX_GEN+1), acum_gbest_vec.mean(axis=0))
    plt.title('Average globally best error across runs')
    plt.xlabel('Generation')
    plt.ylabel('Error [%]')
    plt.show()

    best_ind = np.argmin(acum_train)
    wbest = acum_wbest[best_ind]
    y_pred = forward(test_x, wbest, threshold)

    print(f"\nTrain acc: {100-acum_train[best_ind]:3.3f}%")
    print(f"Test acc: {((1-criterion(y_pred, test_y))[0,0]*100):3.3f}%")
