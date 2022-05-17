import matplotlib.pyplot as plt
from IPython import display

plt.ion()

#función para inicializar la gráfica
def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Entrenando...')
    plt.xlabel('Número de juegos')
    plt.ylabel('Puntuación')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show()
    plt.pause(.1)