import matplotlib.pyplot as plt
import pandas as pd

def alpha(method, title, y='iter of alpha', s=1):
    fig = plt.figure()

    data = pd.read_csv('./{}.csv'.format(method))
    # fig.suptitle(title)

    plt.scatter(data['iter'], data[y], s=s)
    plt.xlabel('iters')
    plt.ylabel('alpha')

    plt.gcf().subplots_adjust(left=0.15)
    plt.savefig('{}_{}.png'.format(method, y), dpi=400)

def grad_at_last(method, title, y='norm of gradient'):
    fig = plt.figure()

    data = pd.read_csv('./{}.csv'.format(method))
    # fig.suptitle(title)

    plt.plot(data['iter'][-30:], data[y][-30:], linewidth=1)
    plt.xlabel('iters')
    plt.ylabel(y)
    # plt.ylim((3000, 180000))

    plt.gcf().subplots_adjust(left=0.15)
    plt.savefig('{}_{}_at last.png'.format(method, y), dpi=400)

def grad_at_first(method, title, y='norm of gradient'):
    fig = plt.figure()

    data = pd.read_csv('./{}.csv'.format(method))
    # fig.suptitle(title)

    plt.plot(data['iter'][:30], data[y][:30], linewidth=1)
    plt.xlabel('iters')
    plt.ylabel(y)
    # plt.ylim((3000, 180000))

    plt.gcf().subplots_adjust(left=0.15)
    plt.savefig('{}_{}_at first.png'.format(method, y), dpi=400)

def grad(method, title, y='norm of gradient'):
    fig = plt.figure()

    data = pd.read_csv('./{}.csv'.format(method))
    # fig.suptitle(title)

    plt.plot(data['iter'], data[y], linewidth=1)
    plt.xlabel('iters')
    plt.ylabel(y)

    plt.gcf().subplots_adjust(left=0.15)
    plt.savefig('{}_{}.png'.format(method, y), dpi=400)

def loss(method, title, y='loss'):
    fig = plt.figure()

    data = pd.read_csv('./{}.csv'.format(method))
    # fig.suptitle(title)

    plt.plot(data['iter'], data[y], linewidth=1)
    plt.xlabel('iters')
    plt.ylabel(y)

    plt.gcf().subplots_adjust(left=0.15)
    plt.savefig('{}_{}.png'.format(method, y), dpi=400)

if __name__ == '__main__':
    loss('GD', 'Loss of Gradient Descent')
    alpha('GD', 'Alpha of Gradient Descent')
    grad_at_first('GD', 'Norm of Gradient of Gradient Descent')
    grad_at_last('GD', 'Norm of Gradient of Gradient Descent')

    loss('NM', 'Loss of Newton Method')
    alpha('NM', 'Alpha of Newton Method', s=10)
    grad('NM', 'Norm of Gradient of Newton Method')

    loss('LIBLINEAR', 'Loss of LIBLINEAR')
    grad('LIBLINEAR', 'Norm of Gradient of LIBLINEAR')
