#dictionary imports
import os
import gym
import numpy as np
from gym import Env, spaces
import pygame
from random import randint, random, choice
import time

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
            pygame.display.set_caption('env')

            self.done = False
            self.clock = pygame.time.Clock()

        #obs space
        self.observation_shape = (151,151,3)
        self.observation_space = spaces.Box(low =0, high = 1, shape = self.observation_shape, dtype=np.float64)

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

            class Laser:
                def __init__(self,axis,pos,time):
                    self.index = 'laser'
                    self.axis = axis
                    self.pos = pos
                    self.initial_time = time
                    self.time = time
                    self.color = (255,255,0)
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
        self.canvas = np.empty(self.observation_shape)*1
        self.canvas[int(self.player.position[0]/4)][int(self.player.position[1]/4)][0] = 1
        for danger in self.board.obstacles[::-1]:
            #if danger.index == 'flame':
            #    if danger.wall%2:
            #        self.canvas[int(danger.getxy()[0]/4)][int(danger.getxy()[1]/4)][1] = danger.time/danger.initial_time
            #    elif not danger.wall%2:
            #        self.canvas[int(danger.getxy()[0]/4)][int(danger.getxy()[1]/4)][2] = danger.time/danger.initial_time
            #elif danger.index == 'bomb':
            #    self.canvas[int(danger.pos[0]/4)][int(danger.pos[1]/4)][3] = danger.time/danger.initial_time
            if danger.index == 'laser':
                if danger.axis:
                    self.canvas[0][int(danger.pos/4)][1] = danger.time/danger.initial_time
                elif not danger.axis:
                    self.canvas[int(danger.pos/4)][0][1] = danger.time/danger.initial_time
            elif danger.index == 'plasma':
                self.canvas[int(danger.pos[0]/4)][int(danger.pos[1]/4)][2] = 1


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
            if random() < (0):
                #500 refers to starting duration of flame, consider replacing with difficulty curve later maybe
                self.board.spawnFlame(randint(0,3),randint(0,600-self.board.flamewidth),500)
            if random() < (0): #
                self.board.spawnLaser(randint(0,1),randint(0,600-self.board.laserwidth),300)
            if random() < (0):
                self.board.spawnBomb(randint(40,560),randint(40,560),250)
            #if random() < (1/350):
            #    self.new_plasmapos = self.board.getplasmapos(randint(1,2320))
            #    self.board.spawnPlasma(self.new_plasmapos[0],self.new_plasmapos[1],[self.player.position[0]-self.new_plasmapos[0],self.player.position[1]-self.new_plasmapos[1]])
            self.new_plasmapos = self.board.getplasmapos(randint(1,2320))
            if len(self.board.obstacles) < 5:
                self.board.spawnPlasma(self.new_plasmapos[0], self.new_plasmapos[1],
                                       [self.player.position[0] - self.new_plasmapos[0],
                                        self.player.position[1] - self.new_plasmapos[1]])



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
                            #self.reward -= (200-(((self.player.position[0]+10 - danger.pos[0]) ** 2 + (self.player.position[1]+10 - danger.pos[1]) ** 2) ** 0.5))/240
                            self.reward -= (200 - (((self.player.position[0] + 10 - danger.pos[0]) ** 2 + (self.player.position[1] + 10 - danger.pos[1]) ** 2) ** 0.5)) / 240

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
                self.reward = -200

            self.past_action = self.current_action



            #Must return obs, reward, done, info
        return self.canvas, self.reward, self.epdone, {}



    def render(self):
        self.screen.fill((0, 0, 0))
        #render obstacles
        for danger in self.board.obstacles[::-1]:
            if danger.index == 'flame':
                if danger.wall%2:
                    pygame.draw.rect(self.screen,danger.color,pygame.Rect(danger.getxy()[0],danger.getxy()[1],self.board.flameheight,self.board.flamewidth))
                else:
                    pygame.draw.rect(self.screen,danger.color,pygame.Rect(danger.getxy()[0], danger.getxy()[1], self.board.flamewidth,self.board.flameheight))
            elif danger.index == 'bomb':
                pygame.draw.circle(self.screen,danger.color,danger.pos,self.board.bombwidth/2)
            elif danger.index == 'laser':
                if danger.axis:
                    pygame.draw.rect(self.screen,danger.color,pygame.Rect(danger.getxy()[0],danger.getxy()[1],self.board.laserwidth,600))
                elif not danger.axis:
                    pygame.draw.rect(self.screen,danger.color,pygame.Rect(danger.getxy()[0],danger.getxy()[1],600,self.board.laserwidth))
            elif danger.index == 'plasma':
                pygame.draw.circle(self.screen,danger.color,(danger.pos[0],danger.pos[1]),self.board.plasmawidth/2)
        #render player
        pygame.draw.rect(self.screen, self.player.rgbcolor,pygame.Rect(self.player.position[0], self.player.position[1], self.player.width,self.player.height))
        pygame.display.update()



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

            class Laser:
                def __init__(self, axis, pos, time):
                    self.index = 'laser'
                    self.axis = axis
                    self.pos = pos
                    self.initial_time = time
                    self.time = time
                    self.color = (255, 255, 0)

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