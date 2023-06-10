def main():
    #dictionary imports
    import os
    import gym
    import numpy as np
    from gym import Env, spaces
    import pygame
    import sys
    from random import randint, random, choice
    import time
    from stable_baselines3 import PPO
    from stable_baselines3 import A2C, DQN
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.evaluation import evaluate_policy

    class StayingAlive(Env):
        def __init__(self, will_render = False):
            #inherits attributes from Env
            super(StayingAlive, self).__init__()

            #regarding simultaneous rendering and running of code
            self.will_render = will_render
            pygame.init()
            self.screen_dimensions = (600, 600)
            if will_render:
                self.screen = pygame.display.set_mode(self.screen_dimensions)
                pygame.display.set_caption('Staying Alive v.1.0')
                pygame.display.set_icon(pygame.image.load('media/Artwork/pdown1.png'))

                self.done = False
                self.clock = pygame.time.Clock()

            #obs space
            self.maxnumofobjects = 5
            self.observation_shape = (5, self.maxnumofobjects, 4)
            # 5 channels, 5 rows, 4 columns
            self.observation_space = spaces.Box(low=-600, high=600, shape=self.observation_shape, dtype=np.uint8)

            #action space
            self.action_space = spaces.Discrete(4)

            self.epdone = False

            class Player:
                def __init__(self):
                    self.width = 30
                    self.height = 30
                    self.position = [300-(self.width/2),300-(self.height/2)]
                    self.speed = 4
                    self.rgbcolor = (255,255,0)

            class Board:
                def __init__(self,width,height):
                    self.width = width
                    self.height = height
                    self.obstacles = []

                    #gameinfo
                    self.flamewidth = 120
                    self.flameheight = 240
                    self.laserwidth = 4
                    self.bombwidth = 80
                    self.plasmawidth= 60

                def getplasmapos(self,pos):
                    if 0 <= pos <= 580:
                        return [pos + 10, 10]
                    elif 580 <= pos <= 1160:
                        return [590, pos - 570]
                    elif 1160 <= pos <= 1740:
                        return [1750 - pos, 590]
                    elif 1740 <= pos <= 2320:
                        return [10, 2330 - pos]

                class Flame:
                    def __init__(self,wall,pos,time):
                        self.index = 'flame'
                        self.wall = wall
                        self.pos = pos
                        self.initial_time = time
                        self.time = time
                        self.color = (255,255,0)
                        self.playedsound = False
                    def getxy(self):
                        if self.wall == 0:
                            return [self.pos,0]
                        elif self.wall == 1:
                            return [360,self.pos]
                        elif self.wall == 2:
                            return [480-self.pos,360]
                        elif self.wall == 3:
                            return [0,480-self.pos]

                class Bomb:
                    def __init__(self,xpos,ypos,time):
                        self.index = 'bomb'
                        self.pos = (xpos,ypos)
                        self.initial_time = time
                        self.time = time
                        self.color = (255,255,0)
                        self.playedsound = False

                class Laser:
                    def __init__(self,axis,pos,time):
                        self.index = 'laser'
                        self.axis = axis
                        self.pos = pos
                        self.initial_time = time
                        self.time = time
                        self.color = (255,255,0)
                        self.playedsound = False
                    def getxy(self):
                        if self.axis == 0:
                            return [0,self.pos]
                        elif self.axis == 1:
                            return [self.pos,0]

                class Plasma:
                    def __init__(self,xpos,ypos,vector):
                        self.index = 'plasma'
                        self.pos = [xpos,ypos]
                        self.vector = vector
                        self.color = (255,0,0)
                        self.time = 1500
                        self.initial_time = 12000
                        self.playedsound = False

                def spawnFlame(self,wall,pos,time):
                    new_flame = self.Flame(wall,pos,time)
                    self.obstacles.append(new_flame)
                def spawnBomb(self,xpos,ypos,time):
                    new_bomb = self.Bomb(xpos,ypos,time)
                    self.obstacles.append(new_bomb)
                def spawnLaser(self,axis,pos,time):
                    new_laser = self.Laser(axis,pos,time)
                    self.obstacles.append(new_laser)
                def spawnPlasma(self,xpos,ypos,vector):
                    new_plasma = self.Plasma(xpos,ypos,vector)
                    self.obstacles.append(new_plasma)





            self.player = Player()
            self.board = Board(self.screen_dimensions[0],self.screen_dimensions[1])
            self.gametime = 0
            self.canvas = np.zeros(self.observation_shape) * 1
            self.reward = -20
            self.past_action = 0

        def draw_elements_on_canvas(self):
            #add playerpos to canvas
            self.canvas = np.zeros(self.observation_shape) * 1
            self.canvas[0:][0][0] = [int(self.player.position[0]), int(self.player.position[1]), int(0), int(0)]
            listofflames = []
            listofbombs = []
            listoflasers = []
            listofplasmas = []
            for danger in self.board.obstacles[::-1]:
                if danger.index == 'flame':
                    listofflames.append(danger)
                elif danger.index == 'bomb':
                    listofbombs.append(danger)
                elif danger.index == 'laser':
                    listoflasers.append(danger)
                elif danger.index == 'plasma':
                    listofplasmas.append(danger)
            for idx, danger in enumerate(listofflames):

                try:
                    if danger.wall % 2:
                        self.canvas[0:][1][idx] = [int(danger.getxy()[0]), int(danger.getxy()[1]),
                                                   int(danger.time / danger.initial_time * 600), int(1)]
                        # column, channel, row
                    else:
                        self.canvas[0:][1][idx] = [int(danger.getxy()[0]), int(danger.getxy()[1]),
                                                   int(danger.time / danger.initial_time * 600), int(2)]
                except:
                    pass
            for idx, danger in enumerate(listofbombs):
                try:
                    self.canvas[0:][2][idx] = [int(danger.pos[0]), int(danger.pos[1]),
                                               int(danger.time / danger.initial_time * 600), 0]
                except:
                    pass
            for idx, danger in enumerate(listoflasers):

                try:
                    self.canvas[0:][3][idx] = [int(danger.getxy()[0]), int(danger.getxy()[1]),
                                               int(danger.time / danger.initial_time * 600), 0]
                except:
                    pass

            for idx, danger in enumerate(listofplasmas):
                try:
                    self.canvas[0:][4][idx] = [int(danger.pos[0]), int(danger.pos[1]), int(danger.vector[0]),
                                               int(danger.vector[1])]
                except:
                    pass

        def step(self, action = 'none'):
            self.current_action = action
            if action == 'none':
                self.current_action = randint(0,3)

            self.reward = -20

            if not self.epdone:

                #regarding movement of player v
                if self.current_action in self.action_space:
                    if self.current_action == 0:
                        self.player.position[1] -= self.player.speed
                        #NOTE THAT THIS LINE FORCES THE ENVIRONMENT INTO TOP LEFT CORNER IN DISPLAY
                        #IF THIS MUST BE CHANGED IN PRESENTATION, CHANGE THIS LINE
                        if self.player.position[1] < 0:
                            self.player.position[1] += self.player.speed
                    elif self.current_action == 1:
                        self.player.position[1] += self.player.speed
                        if self.player.position[1]+self.player.height > self.board.height:
                            self.player.position[1] -= self.player.speed
                    elif self.current_action == 2:
                        self.player.position[0] -= self.player.speed
                        if self.player.position[0] < 0:
                            self.player.position[0] += self.player.speed
                    elif self.current_action == 3:
                        self.player.position[0] += self.player.speed
                        if self.player.position[0] + self.player.width > self.board.width:
                            self.player.position[0] -= self.player.speed


                #obstacle spawning
                if random() < (1/270):
                    #500 refers to starting duration of flame, consider replacing with difficulty curve later maybe
                    self.board.spawnFlame(randint(0,3),randint(0,600-self.board.flamewidth),500)
                if random() < (1/350):
                    self.board.spawnLaser(randint(0,1),randint(0,600-self.board.laserwidth),300)
                if random() < (1/180):
                    self.board.spawnBomb(randint(40,560),randint(40,560),250)
                if random() < (1/350):
                    self.new_plasmapos = self.board.getplasmapos(randint(1,2320))
                    self.board.spawnPlasma(self.new_plasmapos[0],self.new_plasmapos[1],[self.player.position[0]-self.new_plasmapos[0],self.player.position[1]-self.new_plasmapos[1]])


                #rules for expiry + colorstages
                for danger in self.board.obstacles:
                    danger.time -= 1
                    if danger.time <= 0:
                        self.board.obstacles.remove(danger)
                    elif danger.index == 'flame' or 'laser' or 'bomb':

                        if danger.time/danger.initial_time <= (1/8):
                            danger.color = (255,0,0)

                            #checks for collisions with player
                            if danger.index == 'flame':
                                if not danger.wall%2:
                                    if (self.player.position[0] + 20 >= danger.getxy()[0] and self.player.position[0] <= danger.getxy()[0]+120) and (self.player.position[1]+20 >= danger.getxy()[1] and self.player.position[1] <= danger.getxy()[1]+240):
                                        self.epdone = True
                                elif danger.wall%2:
                                    if (self.player.position[0] + 20 >= danger.getxy()[0] and self.player.position[0] <= danger.getxy()[0]+240) and (self.player.position[1]+20 >= danger.getxy()[1] and self.player.position[1] <= danger.getxy()[1]+120):
                                        self.epdone = True
                            if danger.index == 'laser':
                                #if danger.axis and (self.player.position[0] <= danger.getxy()[0]+4 and self.player.position[0] >= danger.getxy()[0]):
                                if danger.axis and ((self.player.position[0] <= danger.getxy()[0] <= self.player.position[0]+20) or (self.player.position[0] <= danger.getxy()[0]+4 <= self.player.position[0]+20)):
                                    self.epdone = True
                                #elif not danger.axis and (self.player.position[1] <= danger.getxy()[1]+4 and self.player.position[1] >= danger.getxy()[1]):
                                elif not danger.axis and ((self.player.position[1] <= danger.getxy()[1] <= self.player.position[1]+20) or (self.player.position[1] <= danger.getxy()[1]+4 <= self.player.position[1]+20)):
                                    self.epdone = True
                            if danger.index == 'bomb':
                                #uses circle equation to detect collision
                                if (self.player.position[0]+10-danger.pos[0])**2 + (self.player.position[1]+10-danger.pos[1])**2 < (self.board.bombwidth/2 + 10)**2:
                                    self.epdone = True


                        elif danger.time/danger.initial_time <= (2/5):
                            danger.color = (255,55,0)

                            if danger.index == 'flame':
                                if not danger.wall%2:
                                    if (self.player.position[0] + 20 >= danger.getxy()[0] and self.player.position[0] <= danger.getxy()[0]+120) and (self.player.position[1]+20 >= danger.getxy()[1] and self.player.position[1] <= danger.getxy()[1]+240):
                                        self.reward -= 0.8
                                elif danger.wall%2:
                                    if (self.player.position[0] + 20 >= danger.getxy()[0] and self.player.position[0] <= danger.getxy()[0]+240) and (self.player.position[1]+20 >= danger.getxy()[1] and self.player.position[1] <= danger.getxy()[1]+120):
                                        self.reward -= 0.8
                            if danger.index == 'laser':
                                #if danger.axis and (self.player.position[0] <= danger.getxy()[0]+4 and self.player.position[0] >= danger.getxy()[0]):
                                if danger.axis and ((self.player.position[0] <= danger.getxy()[0] <= self.player.position[0]+20) or (self.player.position[0] <= danger.getxy()[0]+4 <= self.player.position[0]+20)):
                                    self.reward -= 0.8
                                #elif not danger.axis and (self.player.position[1] <= danger.getxy()[1]+4 and self.player.position[1] >= danger.getxy()[1]):
                                elif not danger.axis and ((self.player.position[1] <= danger.getxy()[1] <= self.player.position[1]+20) or (self.player.position[1] <= danger.getxy()[1]+4 <= self.player.position[1]+20)):
                                    self.reward -= 0.8
                            if danger.index == 'bomb':
                                #uses circle equation to detect collision
                                if (self.player.position[0]+10-danger.pos[0])**2 + (self.player.position[1]+10-danger.pos[1])**2 < (self.board.bombwidth/2 + 10)**2:
                                    self.reward -= 0.8

                        elif danger.time/danger.initial_time <= (3/4):
                            danger.color = (255,155,0)

                            if danger.index == 'flame':
                                if not danger.wall%2:
                                    if (self.player.position[0] + 20 >= danger.getxy()[0] and self.player.position[0] <= danger.getxy()[0]+120) and (self.player.position[1]+20 >= danger.getxy()[1] and self.player.position[1] <= danger.getxy()[1]+240):
                                        self.reward -= 0.4
                                elif danger.wall%2:
                                    if (self.player.position[0] + 20 >= danger.getxy()[0] and self.player.position[0] <= danger.getxy()[0]+240) and (self.player.position[1]+20 >= danger.getxy()[1] and self.player.position[1] <= danger.getxy()[1]+120):
                                        self.reward -= 0.4
                            if danger.index == 'laser':
                                #if danger.axis and (self.player.position[0] <= danger.getxy()[0]+4 and self.player.position[0] >= danger.getxy()[0]):
                                if danger.axis and ((self.player.position[0] <= danger.getxy()[0] <= self.player.position[0]+20) or (self.player.position[0] <= danger.getxy()[0]+4 <= self.player.position[0]+20)):
                                    self.reward -= 0.4
                                #elif not danger.axis and (self.player.position[1] <= danger.getxy()[1]+4 and self.player.position[1] >= danger.getxy()[1]):
                                elif not danger.axis and ((self.player.position[1] <= danger.getxy()[1] <= self.player.position[1]+20) or (self.player.position[1] <= danger.getxy()[1]+4 <= self.player.position[1]+20)):
                                    self.reward -= 0.4
                            if danger.index == 'bomb':
                                #uses circle equation to detect collision
                                if (self.player.position[0]+10-danger.pos[0])**2 + (self.player.position[1]+10-danger.pos[1])**2 < (self.board.bombwidth/2 + 10)**2:
                                    self.reward -= 0.4
                        else:
                            danger.color = (255,255,0)

                            if danger.index == 'flame':
                                if not danger.wall%2:
                                    if (self.player.position[0] + 20 >= danger.getxy()[0] and self.player.position[0] <= danger.getxy()[0]+120) and (self.player.position[1]+20 >= danger.getxy()[1] and self.player.position[1] <= danger.getxy()[1]+240):
                                        self.reward -= 0.2
                                elif danger.wall%2:
                                    if (self.player.position[0] + 20 >= danger.getxy()[0] and self.player.position[0] <= danger.getxy()[0]+240) and (self.player.position[1]+20 >= danger.getxy()[1] and self.player.position[1] <= danger.getxy()[1]+120):
                                        self.reward -= 0.2
                            if danger.index == 'laser':
                                #if danger.axis and (self.player.position[0] <= danger.getxy()[0]+4 and self.player.position[0] >= danger.getxy()[0]):
                                if danger.axis and ((self.player.position[0] <= danger.getxy()[0] <= self.player.position[0]+20) or (self.player.position[0] <= danger.getxy()[0]+4 <= self.player.position[0]+20)):
                                    self.reward -= 0.2
                                #elif not danger.axis and (self.player.position[1] <= danger.getxy()[1]+4 and self.player.position[1] >= danger.getxy()[1]):
                                elif not danger.axis and ((self.player.position[1] <= danger.getxy()[1] <= self.player.position[1]+20) or (self.player.position[1] <= danger.getxy()[1]+4 <= self.player.position[1]+20)):
                                    self.reward -= 0.2
                            if danger.index == 'bomb':
                                #uses circle equation to detect collision
                                if (self.player.position[0]+10-danger.pos[0])**2 + (self.player.position[1]+10-danger.pos[1])**2 < (self.board.bombwidth/2 + 10)**2:
                                    self.reward -= 0.2

                for danger in self.board.obstacles:
                    if danger.index == 'plasma':

                        #collision
                        if (self.player.position[0]+10 - danger.pos[0]) ** 2 + (self.player.position[1]+10-danger.pos[1]) ** 2 < (self.board.plasmawidth/2 + 12) ** 2:
                            self.epdone = True
                        else:
                            if ((self.player.position[0]+10 - danger.pos[0]) ** 2 + (self.player.position[1]+10 - danger.pos[1]) ** 2) ** 0.5 < 200:
                                self.reward -= (200-(((self.player.position[0]+10 - danger.pos[0]) ** 2 + (self.player.position[1]+10 - danger.pos[1]) ** 2) ** 0.5))/240

                        #expiry
                        if (danger.pos[0] < 10 or danger.pos[0] > 590) or (danger.pos[1] < 10 or danger.pos[1] > 590):
                            self.board.obstacles.remove(danger)

                        #movement
                        danger.pos[0] += danger.vector[0] / 200
                        danger.pos[1] += danger.vector[1] / 200


                self.gametime += 1
                if not self.epdone:
                    self.draw_elements_on_canvas()
                    self.reward += 20.5
                if self.epdone:
                    self.reward = -20
                if self.current_action == self.past_action:
                    self.reward += 0.05
                self.past_action = self.current_action



                #Must return obs, reward, done, info
            return self.canvas, self.reward, self.epdone, []



        def render(self):
            #render obstacles
            for danger in self.board.obstacles[::-1]:
                if danger.index == 'flame':

                #    if danger.wall%2:
                #        pygame.draw.rect(env.screen,danger.color,pygame.Rect(danger.getxy()[0],danger.getxy()[1],self.board.flameheight,self.board.flamewidth))
                #    else:
                #        pygame.draw.rect(env.screen,danger.color,pygame.Rect(danger.getxy()[0], danger.getxy()[1], self.board.flamewidth,self.board.flameheight))
                    pass
                elif danger.index == 'bomb':
                #    pygame.draw.circle(env.screen,danger.color,danger.pos,self.board.bombwidth/2)
                    pass
                elif danger.index == 'laser':
                    #if danger.axis:
                    #    pygame.draw.rect(env.screen,danger.color,pygame.Rect(danger.getxy()[0],danger.getxy()[1],self.board.laserwidth,600))
                    #elif not danger.axis:
                    #   pygame.draw.rect(env.screen,danger.color,pygame.Rect(danger.getxy()[0],danger.getxy()[1],600,self.board.laserwidth))
                    pass
                elif danger.index == 'plasma':
                    #pygame.draw.circle(env.screen,danger.color,(danger.pos[0],danger.pos[1]),self.board.plasmawidth/2)
                    pass
            #render player
            #pygame.draw.rect(env.screen, self.player.rgbcolor,pygame.Rect(env.player.position[0], env.player.position[1], env.player.width,env.player.height))




        def reset(self):
            self.screen_dimensions = (600, 600)
            if self.will_render:
                self.screen = pygame.display.set_mode(self.screen_dimensions)
                pygame.display.set_caption('env')

                self.done = False
                self.clock = pygame.time.Clock()

            self.epdone = False

            class Player:
                def __init__(self):
                    self.width = 30
                    self.height = 30
                    self.position = [300 - (self.width / 2), 300 - (self.height / 2)]
                    self.speed = 4
                    self.rgbcolor = (255, 255, 0)

            class Board:
                def __init__(self, width, height):
                    self.width = width
                    self.height = height
                    self.obstacles = []

                    # gameinfo
                    self.flamewidth = 120
                    self.flameheight = 240
                    self.laserwidth = 4
                    self.bombwidth = 80
                    self.plasmawidth = 60

                def getplasmapos(self, pos):
                    if 0 <= pos <= 580:
                        return [pos + 10, 10]
                    elif 580 <= pos <= 1160:
                        return [590, pos - 570]
                    elif 1160 <= pos <= 1740:
                        return [1750 - pos, 590]
                    elif 1740 <= pos <= 2320:
                        return [10, 2330 - pos]

                class Flame:
                    def __init__(self, wall, pos, time):
                        self.index = 'flame'
                        self.wall = wall
                        self.pos = pos
                        self.initial_time = time
                        self.time = time
                        self.color = (255, 255, 0)
                        self.playedsound = False

                    def getxy(self):
                        if self.wall == 0:
                            return [self.pos, 0]
                        elif self.wall == 1:
                            return [360, self.pos]
                        elif self.wall == 2:
                            return [480 - self.pos, 360]
                        elif self.wall == 3:
                            return [0, 480 - self.pos]

                class Bomb:
                    def __init__(self, xpos, ypos, time):
                        self.index = 'bomb'
                        self.pos = (xpos, ypos)
                        self.initial_time = time
                        self.time = time
                        self.color = (255, 255, 0)
                        self.playedsound = False

                class Laser:
                    def __init__(self, axis, pos, time):
                        self.index = 'laser'
                        self.axis = axis
                        self.pos = pos
                        self.initial_time = time
                        self.time = time
                        self.color = (255, 255, 0)
                        self.playedsound = False

                    def getxy(self):
                        if self.axis == 0:
                            return [0, self.pos]
                        elif self.axis == 1:
                            return [self.pos, 0]

                class Plasma:
                    def __init__(self, xpos, ypos, vector):
                        self.index = 'plasma'
                        self.pos = [xpos, ypos]
                        self.vector = vector
                        self.color = (255, 0, 0)
                        self.time = 1500
                        self.initial_time = 12000
                        self.playedsound = False

                def spawnFlame(self, wall, pos, time):
                    new_flame = self.Flame(wall, pos, time)
                    self.obstacles.append(new_flame)

                def spawnBomb(self, xpos, ypos, time):
                    new_bomb = self.Bomb(xpos, ypos, time)
                    self.obstacles.append(new_bomb)

                def spawnLaser(self, axis, pos, time):
                    new_laser = self.Laser(axis, pos, time)
                    self.obstacles.append(new_laser)

                def spawnPlasma(self, xpos, ypos, vector):
                    new_plasma = self.Plasma(xpos, ypos, vector)
                    self.obstacles.append(new_plasma)

            self.player = Player()
            self.board = Board(self.screen_dimensions[0], self.screen_dimensions[1])
            self.gametime = 0
            self.canvas = np.zeros(self.observation_shape) * 1
            self.draw_elements_on_canvas()
            self.reward = -20
            self.past_action = 0

            return self.canvas
            #May need to double check in case this causes any problems, but it seems like it works so far?

        def close(self):
            self.done = True


    #Instance of env class
    env = StayingAlive(True)
    basedir = 'Agent_Directory/savedmodels/'
    model = PPO.load(basedir + 'agent_in_full_environment_PPO', env, verbose = 1)

    if env.will_render:
        userinput = 4

        #load media
        artdir = "media/Artwork/"
        audir = "media/Audio/"
        #all artwork is original
        background_png = pygame.image.load(artdir + "background.png")
        scorecard_png = pygame.image.load(artdir + "scorecard.png")
        player_costumes = [[pygame.image.load(artdir + "pup1.png"),
                            pygame.image.load(artdir + "pup2.png")],
                           [pygame.image.load(artdir + "pdown1.png"),
                            pygame.image.load(artdir + "pdown2.png")],
                           [pygame.image.load(artdir + "pleft1.png"),
                            pygame.image.load(artdir + "pleft2.png")],
                           [pygame.image.load(artdir + "pright1.png"),
                            pygame.image.load(artdir+"pright2.png")],
                           [pygame.image.load(artdir + "pdown1.png"),
                            pygame.image.load(artdir + "pdown1.png")]]
        player_death = [pygame.image.load(artdir + "explode1.png"),
                        pygame.image.load(artdir + "explode2.png"),
                        pygame.image.load(artdir + "explode3.png"),
                        pygame.image.load(artdir + "explode4.png"),
                        pygame.image.load(artdir + "explode5.png"),
                        pygame.image.load(artdir + "explode6.png"),
                        pygame.image.load(artdir + "explode7.png")]
        nflame = [[pygame.image.load(artdir + "nflame1.png"),pygame.image.load(artdir + "null.png")],
                  [pygame.image.load(artdir + "nflame2.png"),pygame.image.load(artdir + "null.png")],
                  [pygame.image .load(artdir + "nflame3.png"),pygame.image.load(artdir + "null.png")]]
        leflame = [pygame.image.load(artdir + "flame1.png"),
                 pygame.image.load(artdir + "flame2.png"),
                 pygame.image.load(artdir + "flame3.png"),
                 pygame.image.load(artdir + "flame4.png"),
                 pygame.image.load(artdir + "flame5.png"),
                 pygame.image.load(artdir + "flame6.png"),
                 pygame.image.load(artdir + "flame5.png"),
                 pygame.image.load(artdir + "flame4.png"),
                 pygame.image.load(artdir + "flame3.png"),
                 pygame.image.load(artdir + "flame2.png"),
                 pygame.image.load(artdir + "flame1.png")]
        lbomb = [[pygame.image.load(artdir + "lbomb1.png"),pygame.image.load(artdir + "null.png")],
                 [pygame.image.load(artdir + "lbomb2.png"),pygame.image.load(artdir + "null.png")],
                 [pygame.image.load(artdir + "lbomb3.png"),pygame.image.load(artdir + "null.png")]]
        plasmaf = [pygame.image.load(artdir + "plasma1.png"),
                   pygame.image.load(artdir + "plasma2.png"),
                   pygame.image.load(artdir + "plasma3.png")]
        nlaser = [[pygame.image.load(artdir + "nlaser1.png"), pygame.image.load(artdir + "null.png")],
                  [pygame.image.load(artdir + "nlaser2.png"), pygame.image.load(artdir + "null.png")],
                  [pygame.image.load(artdir + "nlaser3.png"), pygame.image.load(artdir + "null.png")]]
        laserf = [pygame.image.load(artdir + "laser1.png"),
                  pygame.image.load(artdir + "laser2.png"),
                  pygame.image.load(artdir + "laser3.png")]
        startscreen = pygame.image.load(artdir + "startscreen.png")
        #sound effects are from https://pixabay.com/sound-effects/search/game/
        sgamestart = pygame.mixer.Sound(audir + "gamestart.mp3")
        sgamestart.set_volume(0.4)
        sdeath = pygame.mixer.Sound(audir + "videogame-death-sound-43894.mp3")
        #https://elements.envato.com/fire-fireball-cast-LQT5FB4?utm_source=mixkit&utm_medium=referral&utm_campaign=elements_mixkit_cs_sfx_tag&_ga=2.139135219.1185745211.1666358135-2045794747.1666358135
        sfire = pygame.mixer.Sound(audir + "fire.mp3")
        sfire.set_volume(0.4)
        sbomb = pygame.mixer.Sound(audir + "boom.mp3")
        sbomb.set_volume(0.15)
        slaser = pygame.mixer.Sound(audir + "pew2.mp3")
        slaser.set_volume(0.4)
        splasma = pygame.mixer.Sound(audir + "zap.mp3")
        splasma.set_volume(0.2)
        pygame.mixer.music.load(audir + "soundtrack.mp3")

        pygame.mixer.music.play(-1)
        pygame.mixer.music.set_volume(0.8)
        pregame = True
        while pregame:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    window_running = False
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        pregame = False
            env.screen.blit(pygame.transform.scale(startscreen,(600,600)), (0,0))
            pygame.display.update()



        window_running = True
        pygame.mixer.Sound.play(sgamestart)
        obs = env.reset()

        while window_running:
            while not env.done:

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.done = True
                        window_running = False
                        #quits by throwing exception?
                        sys.exit()


                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_UP:
                            userinput = 0
                        elif event.key == pygame.K_DOWN:
                            userinput = 1
                        elif event.key == pygame.K_LEFT:
                            userinput = 2
                        elif event.key == pygame.K_RIGHT:
                            userinput = 3


                env.clock.tick(48)

                env.screen.blit(background_png, (0,-36))

                env.render()
                colorindex = [(255,255,0),(255,155,0),(255,55,0),(255,0,0)]
                #obstacle render
                for danger in env.board.obstacles[::-1]:
                    if danger.index == 'flame':
                        if colorindex.index(danger.color) in (0,1,2):
                            if danger.wall == 0:
                                env.screen.blit(pygame.transform.rotate(nflame[colorindex.index(danger.color)][int((danger.time/32)%2)], 180), (danger.getxy()[0], danger.getxy()[1]))
                            elif danger.wall == 1:
                                env.screen.blit(pygame.transform.rotate(nflame[colorindex.index(danger.color)][int((danger.time/32)%2)], 90), (danger.getxy()[0], danger.getxy()[1]))
                            elif danger.wall == 2:
                                env.screen.blit(nflame[colorindex.index(danger.color)][int((danger.time/32)%2)],(danger.getxy()[0], danger.getxy()[1]))
                            elif danger.wall == 3:
                                env.screen.blit(pygame.transform.rotate(nflame[colorindex.index(danger.color)][int((danger.time/32)%2)], 270), (danger.getxy()[0], danger.getxy()[1]))
                        else:
                            if not danger.playedsound:
                                danger.playedsound = True
                                pygame.mixer.Sound.play(sfire)
                            if danger.wall == 0:
                                env.screen.blit(pygame.transform.rotate(leflame[int((danger.time/danger.initial_time*1000) / 12)], 180), (danger.getxy()[0],danger.getxy()[1]))
                            elif danger.wall == 1:
                                env.screen.blit(pygame.transform.rotate(leflame[int((danger.time / danger.initial_time * 1000) / 12)], 90),(danger.getxy()[0], danger.getxy()[1]))
                            elif danger.wall == 2:
                                env.screen.blit(pygame.transform.rotate(leflame[int((danger.time / danger.initial_time * 1000) / 12)], 0),(danger.getxy()[0], danger.getxy()[1]))
                            elif danger.wall == 3:
                                env.screen.blit(pygame.transform.rotate(leflame[int((danger.time / danger.initial_time * 1000) / 12)], 270),(danger.getxy()[0], danger.getxy()[1]))
                    elif danger.index == "bomb":
                        if colorindex.index(danger.color) in (0,1,2):
                            env.screen.blit(lbomb[colorindex.index(danger.color)][int((danger.time/16)%2)], (danger.pos[0]-40, danger.pos[1]-40))
                        else:
                            if not danger.playedsound:
                                danger.playedsound = True
                                pygame.mixer.Sound.play(sbomb)
                            env.screen.blit(pygame.transform.scale(player_death[int((danger.time / danger.initial_time * 1000) / 18)],(80,80)), (danger.pos[0]-40, danger.pos[1]-40))
                    elif danger.index == "plasma":
                        if not danger.playedsound:
                            pygame.mixer.Sound.play(splasma)
                            danger.playedsound = True

                        env.screen.blit(plasmaf[int(danger.time/12)%3],(danger.pos[0]-30,danger.pos[1]-30))
                    elif danger.index == "laser":
                        if colorindex.index(danger.color) in (0, 1, 2):
                            if danger.axis:
                                env.screen.blit(nlaser[colorindex.index(danger.color)][int(danger.time/32)%2], (danger.getxy()[0]-3,danger.getxy()[1]))
                            elif not danger.axis:
                                env.screen.blit(pygame.transform.rotate(nlaser[colorindex.index(danger.color)][int(danger.time/32)%2],90), (danger.getxy()[0], danger.getxy()[1]-3))
                        else:
                            if not danger.playedsound:
                                pygame.mixer.Sound.play(slaser)
                                danger.playedsound = True
                            if danger.axis:
                                env.screen.blit(laserf[int((danger.time/danger.initial_time * 1000)/8)%3], (danger.getxy()[0]-3, danger.getxy()[1]))
                            elif not danger.axis:
                                env.screen.blit(pygame.transform.rotate(laserf[int((danger.time/danger.initial_time * 1000)/8)%3], 90), (danger.getxy()[0], danger.getxy()[1]-3))
                #player render
                env.screen.blit(player_costumes[userinput][int((env.gametime/5)%2)],(env.player.position[0],env.player.position[1]))

                #score display
                font = pygame.font.Font('media/Font/gameria.ttf',32)
                text = font.render(str(env.gametime),True,(100,100,255))
                env.screen.blit(scorecard_png, (0, 0))
                env.screen.blit(text,(10,10))

                pygame.display.update()
                #for temporary purposes
                userinput, _states = model.predict(obs)

                obs, reward, env.done, info = env.step(userinput)

            pygame.mixer.Sound.play(sdeath)
            inbetween = True
            posttime = 0
            while inbetween:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        window_running = False
                        sys.exit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            inbetween = False
                            env.done = False
                            env.reset()
                env.screen.fill((0,0,0))
                if posttime < 54:
                    env.screen.blit(player_death[int(posttime/8)],(env.player.position[0],env.player.position[1]))
                else:
                    env.screen.blit(player_death[6], (env.player.position[0], env.player.position[1]))

                font = pygame.font.Font('media/Font/gameria.ttf', 64)
                text = font.render(f"You scored", True, (100, 100, 255))
                env.screen.blit(text, (10, 10))
                text = font.render(str(env.gametime), True, (80,80,205))
                env.screen.blit(text, (40,100))
                font = pygame.font.Font('media/Font/gameria.ttf', 32)
                text = font.render("Press space to play again", True, (100,100,255))
                env.screen.blit(text, (10,400))

                pygame.display.update()
                posttime += 1
                env.clock.tick(48)
            pygame.mixer.Sound.play(sgamestart)

    else:

        pass


if __name__ == '__main__':
    main()

