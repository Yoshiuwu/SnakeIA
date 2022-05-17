import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

# Inicializamos Pygame
pygame.init()

# Letras utilizadas en la interfaz
font = pygame.font.SysFont('arial', 25)

# Movimientos que puede hacer snake
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Tupla con las coordenadas de la pantalla
Point = namedtuple('Point', 'x, y')

# Colores utilizados en el programa
WHITE = (255, 255, 255)
RED = (200,0,0)
GREEN = (0, 143, 57)
BLACK = (0,0,0)

# Tamaño de los bloques dentro del programa
BLOCK_SIZE = 20
# Velocidad de snake
SPEED = 40

class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # Inicialización de dysplay
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # Estado inicial del juego
        self.direction = Direction.RIGHT

        # Tamaño inicial de snake
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        # Inicialización de variables necesarias
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    # Coloca la comida en una coordenada
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        # Obtiene una coordenada aleatoria
        self.food = Point(x, y)
        # Validación para que no se imprima en el lugar donde se encuentra snake
        if self.food in self.snake:
            self._place_food()

    # Eventos dentro del juego
    def play_step(self, action):
        self.frame_iteration += 1
        # 1. Primer frame y opción de salir
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. Movimiento
        self._move(action) # Actualización de la cabeza
        self.snake.insert(0, self.head)
        
        # 3. Revisar si ya perdió
        reward = 0
        game_over = False
        # Si se colisiona o se cicla el movimiento se determina el fin del juego
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            # Castigo por perder el juego
            reward = -10
            return reward, game_over, self.score

        # 4. Cuando come se le da un punto y un premio
        #    para reforzar el aprendizaje
        if self.head == self.food:
            self.score += 1
            reward = 10

            # Se vuelve a colocar la comida
            self._place_food()
        else:
            # Si no come se actualiza la cola
            self.snake.pop()
        
        # 5. Se actualiza la UI y el reloj del juego
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. Al terminar se regresan los resultados obtenidos del movimiento
        return reward, game_over, self.score

    # Determina si hay colisión
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Si colisiona con la pantalla
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Si colisiona con su cuerpo
        if pt in self.snake[1:]:
            return True
        # En caso de nos ser así, se determina que no existe colisión
        return False

    # Actualizar UI
    def _update_ui(self):
        # Llena el fondo de color negro
        self.display.fill(BLACK)
        
        # Dibuja snake
        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))

        # Dibuja la comida
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # La puntuación se muestra en la UI
        text = font.render("Puntuación: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    # Definir los tipos de movimientos
    def _move(self, action):
        # Posibles movimientos
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        # No hay cambios en el movimiento
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        # Gira en la dirección de las manecillas del reloj
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        # Gira en el sentido contrario a las manecillas
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        # Determina la dirección que va a tomar
        self.direction = new_dir

        #Se dan las coordenadas y se mueve
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        #Se mueve a la nueva coordenada
        self.head = Point(x, y)