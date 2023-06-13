import DQN
import pygame
from layers import *
import random
import numpy as np
import math
import time


"""
Environment is a NxN world that has organisms on it
Food randomly spawns in the world


"""



LR = 0.0000001
FOOD_REWARD = 1
HIDDEN_LAYERS = [

]

WINDOW_SIZE = (700, 700)
STEPS_UNTIL_DISPLAY = 20000

pygame.init()
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption('Simulation')

# screen = None
# pygame.display.set_caption('Simulation')

def main():

    env = Environment(screen, (10, 10), 70, display=True)
    env.spawn_food(100)
    env.spawn_organisms(1)
    
    running = True
    step = 0
    while running:
        step += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        # time.sleep(.02)
        env.step()
        print(f'Step: {step}')
        # if step > 1000:
        #     env.display = True

class Environment:
    def __init__(self, screen, num_chunks, chunk_size, display=True) -> None:
        self.orgs = []
        self.display=display
        self.screen = screen
        self.num_chunks = num_chunks
        self.chunk_size = chunk_size
        self.size = (num_chunks[0] * chunk_size, num_chunks[1] * chunk_size)
        self.chunks = {(x,y):Chunk(x, y) for x in range(num_chunks[0]) 
                                         for y in range(num_chunks[1])}
        self.food_size = 5 # radius
        self.food_energy = 25
        self.food_reward = 10
        self.food_color = (255,255,255)
        self.n_food = 0
        self.max_food = 100
        self.time = 0


    def spawn_organisms(self, n):
        for ID in range(n):
            o = Organism(lr=LR, ID=ID, vision=8, env=self, pos=self.get_random_pos())
            self.orgs.append(o)
        pass

    
    def spawn_food(self, n):
        for _ in range(n):
            # if self.n_food > self.max_food:
            #     return
            # self.n_food += 1
            pos = self.get_random_pos()
            chunk = self.chunks[get_chunk(pos, self.chunk_size)]
            chunk.add_food(pos)


    def step(self):
        self.time += 1
        for org in self.orgs:
            org.step()
        # if self.time % 500 == 0:
        #     self.spawn_food(20)  
        if self.display:
            self.draw()
  

    def get_random_pos(self):
        return (random.randrange(0, self.size[0]), random.randrange(0, self.size[1]))
    
    def get_random_chunk_pos(self):
        return (random.randrange(0, self.chunk_size), random.randrange(0, self.chunk_size))
    
    def draw(self):
        self.screen.fill("Black")
        for coords, chunk in self.chunks.items():
            # print(coords, chunk)
            for org in chunk.orgs:
                org.draw(self.screen)
            for food in chunk.food:
                pygame.draw.circle(self.screen, self.food_color, (food[0], food[1]), self.food_size)
        pygame.display.update()


    def calc_reward(self, food_eaten):
        return food_eaten * self.food_reward


# ===================================================
#                       ORGANISM
# ===================================================




class Organism:
    def __init__(self, lr, ID, env, pos=(0,0), size=10, vision=5, FOV = math.pi/2) -> None:
        self.actions = [self.forward, self.turn_left, self.turn_right]

        layers = [Dense((len(self.actions),), 'linear', input_size=(vision,))]
        self.agent = DQN.DoubleDQNAgent(n_actions=len(self.actions), layers=layers, lr=lr, epsilon_decay=.999, discount_factor=.99)

        self.env = env

        self.pos = pos
        self.x = pos[0]
        self.y = pos[1]
        self.chunk_pos = get_chunk(self.pos, env.chunk_size)
        self.chunk = env.chunks[self.chunk_pos]
        self.chunk.add_org(self)
        self.angle = 0 # in radians

        self.ID = ID
        # self.color = (255*ID, 40, 40)
        self.color = (255, 0, 0)

        self.n_children = 0
        self.age = 0
        self.energy = 100
        self.energy_decay = 1
        self.max_speed = 10
        self.speed = 5
        self.speed_increment = .50 # increase/decrease speed by %
        self.turn_speed = .1 * math.pi # in radians
        self.size = size # radius
        self.views = vision
        self.FOV = FOV

        self.step_memory = [self.observe(), 0, 0, None, False] # memory of the current step


    def step(self):
        # state, action, reward, next_state, terminal
        observation = self.observe()
        # finish previous memory and store it
        self.step_memory[3] = observation
        self.remember()
        
        # start new memory
        self.step_memory[0] = observation
        action = self.act(observation)
        self.step_memory[1] = action

        food_eaten = self.check_food_collision()
        self.step_memory[2] = self.env.calc_reward(food_eaten)
        self.step_memory[3] = None # will be added later
        self.step_memory[4] = self.check_death()

        self.train()


    def observe(self):
        collisions = np.array([self.env.chunk_size * 2] * self.views)
        for x in range(-1, 2):
            for y in range(-1, 2):
                if self.chunk.x + x < 0 \
                or self.chunk.x + x >= self.env.num_chunks[0] \
                or self.chunk.y + y < 0 \
                or self.chunk.y + y >= self.env.num_chunks[1]:
                    continue
                chunk = self.env.chunks[(self.chunk.x + x, self.chunk.y + y)]
                # for o in chunk.orgs:
                #     if o.ID != self.ID:
                #         raycast(collisions, self.x - o.x, self.y - o.y, o.size, self.angle, self.views, self.FOV, 0) # 0 is organism
                for f in chunk.food:
                    raycast(collisions, self.x, self.y, f[0], f[1], self.env.food_size, self.angle, self.views, self.FOV, 1) # 1 is food
        
        # self.step_memory[0] = collisions
        return collisions


    def act(self, observation):
        action = self.agent.act(observation)
        self.actions[action]()
        return action


    def forward(self):
        # update position
        self.x += int(self.speed * math.cos(abs(self.angle)))
        self.y += int(self.speed * math.sin(abs(self.angle)))

        # wrap edges
        if self.x >= self.env.size[0]:
            self.x -= self.env.size[0]
        elif self.x < 0:
            self.x += self.env.size[0]

        if self.y >= self.env.size[1]:
            self.y -= self.env.size[1]
        elif self.y < 0:
            self.y += self.env.size[1]

        self.pos = (self.x, self.y)

        # update chunk
        new_chunk_pos = get_chunk(self.pos, self.env.chunk_size)
        if new_chunk_pos != self.chunk_pos:
            self.chunk.remove_org(self)
            self.chunk_pos = new_chunk_pos
            self.chunk = self.env.chunks[new_chunk_pos]
            self.chunk.add_org(self)

        # lose energy
        self.energy -= self.speed * self.speed/(self.speed+1) + 1 * self.energy_decay
        self.check_food_collision()


    def turn_right(self):
        self.angle += self.turn_speed
        self.angle %= 2 * math.pi

    def turn_left(self):
        self.angle -= self.turn_speed
        self.angle %= 2 * math.pi

    def increase_speed(self):
        # if self.speed == 0:
        #     self.speed = 1
        # self.speed *= 1 + self.speed_increment
        # self.speed = min(self.max_speed, self.speed)
        self.speed = min(self.max_speed, self.speed + 1)
    
    def decrease_speed(self):
        # self.speed *= 1 - self.speed_increment
        self.speed = max(0, self.speed - 1)

    def reproduce(self):
        self.n_children += 1
        mut_lr = self.lr + np.random.normal(scale=.01)
        mut_size = min(1, self.size + np.random.normal(scale=0.1))

        if random.random() < self.env.mut_rate:
            mut_views = random.randint(-1, 1)
        else:
            mut_views = self.views

        mut_FOV = self.FOV + np.random.normal(scale=.1)
        new_ID = f'{self.ID}-{self.n_children}'
        
        child = Organism(mut_lr, self.n_actions, new_ID, self.env, self.x, self.y, mut_size, mut_views, mut_FOV)
        self.chunk.add_org(child)


    def check_death(self):
        if self.energy <= 0:
            # self.die()
            return True
        return False


    def die(self):
        self.chunk.remove_org(self)
    
    def eat(self, chunk, food):
        # self.env.n_food -= 1
        self.env.spawn_food(1)

        chunk.remove_food(food)
        
        self.energy += self.env.food_energy

    def check_food_collision(self):
        n = 0
        for x in range(-1, 2):
            for y in range(-1, 2):
                if self.chunk.x + x < 0 \
                or self.chunk.x + x >= self.env.num_chunks[0] \
                or self.chunk.y + y < 0 \
                or self.chunk.y + y >= self.env.num_chunks[1]:
                    continue
                chunk = self.env.chunks[(x + self.chunk.x, y + self.chunk.y)]
                eaten = []
                for food in chunk.food:
                    # print(dist(food, (self.x, self.y)) < 20)
                    if dist(food, (self.x, self.y)) <= self.env.food_size + self.size:
                        eaten.append(food)
                for food in eaten:
                    self.eat(chunk, food)
                    n += 1
        return n
    

    def remember(self):
        # state, action, reward, next_state, terminal
        self.agent.remember(*self.step_memory)
    
    
    def train(self):
        self.agent.train(16)

    
    def draw(self, screen):
        pygame.draw.circle(screen, self.color, self.pos, self.size)
        # pygame.draw.aaline(screen, "red", (self.pos), 
        #                   (self.x + int(self.size * math.cos(abs(self.angle))), 
        #                    self.y + int(self.size * math.sin(abs(self.angle)))))
        
        # angles = [self.angle - self.FOV/2 + self.FOV/self.views * i for i in range(self.views)]
        # for angle in angles:
        #     endpoint = (self.x + int(100 * math.cos(abs(angle))), 
        #                 self.y + int(100 * math.sin(abs(angle))))
        #     pygame.draw.aaline(screen, "blue", self.pos, endpoint)

    

def raycast(observations, start_x, start_y, x, y, radius, dir, n, FOV, obj):
    # check for collisions using n rays from -FOV/2 to FOV/2
    # updates the observations list in place
    angles = [(dir - FOV/2 + FOV/n * i) for i in range(n)]
    # for angle in angles:
    #     pygame.draw.aaline(screen, "red", (start_x, start_y), (start_x + 100*math.cos(angle), start_y + 100*math.sin(angle)))
    #     pygame.display.update()
    # for angle, idx in zip(range(int(dir - FOV/2), int(dir + FOV/2), int(FOV/n)), range(n)):
    for idx, angle in enumerate(angles):
        collision = check_collision(angle, start_x, start_y, x, y, radius)
        if collision < observations[idx]:
            observations[idx] = collision
            # pygame.draw.aaline(screen, "red", (start_x, start_y), (start_x + 100*math.cos(angle), start_y + 100*math.sin(angle)))
            # pygame.display.update()
            pass
        else:
            # pygame.draw.aaline(screen, "blue", (start_x, start_y), (start_x + 100*math.cos(angle), start_y + 100*math.sin(angle)))
            # pygame.display.update()
            pass


def check_collision(angle, start_x, start_y, x, y, radius):
    # returns the distance between a ray origin and circle if they collide, or 500 if they don't

    if angle == math.pi/2:
        # looking straight up
        if y < start_y:
            # can't see if below
            return 500

    elif angle == 3*math.pi / 2:
        # looking straight down
        if y > start_y:
            # can't see if above
            return 500

    elif angle < math.pi/2 or angle > 3*math.pi / 2:
        # looking at right side
        if x < start_x:
            return 500
    else:
        # looking at left side
        if x > start_x:
            return 500 

    # find distance from line to circle
    # abs(a*x + b*y + c) / sqrt(a**2 + b**2) regular equation
    d = abs(math.cos(angle) * (start_y - y) - math.sin(angle) * (start_x - x))
    if d > radius:
        return 500
    return math.sqrt((x - start_x)**2 + (y - start_y)**2)




class Chunk:
    def __init__(self, x, y):
        self.coords = (x, y)
        self.x = x
        self.y = y
        self.orgs = set()
        self.food = set()
    
    def add_org(self, org):
        self.orgs.add(org)
    
    def remove_org(self, org):
        self.orgs.remove(org)
    
    def add_food(self, food):
        self.food.add(food)
    
    def remove_food(self, food):
        self.food.remove(food)
        pass


# class Point:
#     def __init__(self, x, y) -> None:
#         self.x = x
#         self.y = y
#         self.pos = (x, y)
    
#     def __init__(self, pos):
#         self.x = pos[0]
#         self.y = pos[1]
#         self.pos = pos


#     def get_x(self):
#         return self.x

#     def get_y(self):
#         return self.y

def get_chunk(pos, chunk_size):
    return (pos[0] // chunk_size, pos[1] // chunk_size)

def dist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx**2 + dy**2)

def dist_squared(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx**2 + dy**2



if __name__ == "__main__":
    main()