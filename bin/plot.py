import matplotlib.pyplot as plt
import pandas as pd

def main(method, title, y):
    fig = plt.figure()

    data = pd.read_csv('./{}.csv'.format(method))
    fig.suptitle(title)

    plt.plot(data['iter'], data[y])
    plt.xlabel('iters')
    plt.ylabel(y)

    plt.gcf().subplots_adjust(left=0.15)
    plt.savefig('{}_{}.png'.format(method, y), dpi=400)

if __name__ == '__main__':
    main('./GD_log', 'Loss of Gradient Descent', 'loss')
    main('./NM_log', 'Loss of Newton Method', 'loss')
