import pygame
import numpy
import random

pygame.init()

WHITE = (0,0,0)
BLACK = (255,255,255)

class Pong:
    def __init__(self):
        self.size = (700, 500)
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("Pong")


        self.screen.fill(BLACK)
        pygame.draw.line(self.screen, WHITE, [349, 0], [349, 500], 5)
        pygame.display.flip()

    def step(p1_action, p2_action):
        pass

a = Pong()