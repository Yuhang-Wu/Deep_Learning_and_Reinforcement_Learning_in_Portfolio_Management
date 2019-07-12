import matplotlib.pyplot as plt


def drawHist(data):
    """ Draw the learning step """

    fig = plt.figure(facecolor='white')
    fig, ax = plt.subplots(2, 1, figsize=(18, 8))

    ax[0].plot(data.loss, label="Loss")
    ax[0].set_title("LOSS")
    ax[0].legend()

    ax[1].plot(data.acc, label="Accuracy")
    ax[1].set_title("ACCURACY")
    ax[1].legend()

    plt.xlabel("Iteration")
    plt.suptitle("Learning Steps", fontsize=18)
    plt.show()
