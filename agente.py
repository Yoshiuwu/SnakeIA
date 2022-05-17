import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from modelo import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # Random
        self.gamma = 0.9 # Loss
        self.memory = deque(maxlen=MAX_MEMORY) # popleft
        self.model = Linear_QNet(11, 256, 3) #Num Neuronas x Capa
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            #Peligro Delante
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            #Peligro Derecha
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            #Peligro izquierda
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            #Movimiento solo decide uno de estos
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            #Identifica la coordenada de la comida
            game.food.x < game.head.x,  # Izquierda
            game.food.x > game.head.x,  # Derecha
            game.food.y < game.head.y,  # Arriba
            game.food.y > game.head.y  # Abajo
            ]
        #retorna el estado del juego
        return np.array(state, dtype=int)
    
    #Recuerda que fue lo último en pasar
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft si se alcanza el tamaño máximo

    #Entrenamiento a largo plazo
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # Lista de tuplas
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    #entrenamiento rápido
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Movimientos aleatorios en la primera iteración
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

#Entrenamiento
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI() 
    #ciclo del juego
    while True:
        # obtener el estado pasado
        state_old = agent.get_state(game)

        # decidir el movimiento
        final_move = agent.get_action(state_old)

        # hacer el movimiento y obtener un nuevo estado
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # entrenamiento rápido
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # recuerda el estado, movimiento, premio y el nuevo estado
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # cuando termina, se reiniciar el juego e inicia el entrenamiento
            game.reset()
            #Con cada iteración aumenta el número de juegos
            agent.n_games += 1
            agent.train_long_memory()
            ###################################################################################################################################
            if agent.n_games > 5:
                game.clock.tick(5)
            #Se se obtiene una puntuación mayor al record
            #en ese entonces se guarda dentro del modelo
            if score > record:
                record = score
                agent.model.save()

            print('Juego', agent.n_games, 'Puntuación', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            #Media de score obtenido y número de juegos
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            #Se muestra la gráfica de scores y número de juegos
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    #Inicia el entrenamiento
    train()