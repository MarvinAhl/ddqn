import pygame
import physics
from rocket import Rocket

from dddqn import DDDQN
import torch
import numpy as np
import matplotlib.pyplot as plt

class Game:
    def __init__(self, render=True, agent_play=True, agent_train=True, agent_file='rocket_game.net', device='cpu'):
        self.running = True
        self.display_surf = None
        self.size = (self.width, self.height) = (1280, 720)

        self.rocket = Rocket()
        self.last_dirty_rects = []

        self.agent_play = agent_play
        self.agent_train = agent_train
        self.agent_file = agent_file

        if agent_play:
            self.agent = DDDQN(6, 4, hidden_layers=(500, 500, 500), gamma=0.99, learning_rate_start=0.0005, learning_rate_decay_steps=50000, learning_rate_min=0.0003,
                               epsilon_start=1.0, epsilon_decay_steps=20000, epsilon_min=0.1, temp_start=10, temp_decay_steps=20000, temp_min=0.1, buffer_size_min=200,
                               buffer_size_max=50000, batch_size=50, replays=1, tau=0.01, alpha=0.6, beta=0.1, beta_increase_steps=20000, device=device)
            if not agent_train:
                self.agent.load_net('rocket_game.net')
        else:
            self.agent = None

        self.render = render

        # For measuring frame time
        self.last_time_ms = 0
        self.updates_per_frame = 100

    def _init(self):
        pygame.init()
        self.display_surf = pygame.display.set_mode(self.size, pygame.SCALED, vsync=1)
        pygame.display.set_caption("Rocket Game")

        self.rocket.init()

    def _on_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
        
        # Controls for human play
        if not self.agent_play:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.rocket.set_f1(1.0)  # Activate Booster
                elif event.key == pygame.K_a:
                    self.rocket.set_theta(1.0)  # Angle Booster left
                elif event.key == pygame.K_d:
                    self.rocket.set_theta(-1.0)  # Angle Booster right
                elif event.key == pygame.K_LEFT:
                    self.rocket.set_f2(-1.0)  # Fire right control nozzle
                elif event.key == pygame.K_RIGHT:
                    self.rocket.set_f2(1.0)  # Fire left control nozzle

            elif event.type == pygame.KEYUP:
                pressed_keys = pygame.key.get_pressed()
                if event.key == pygame.K_SPACE:
                    self.rocket.set_f1(0.0)  # Booster engine off
                elif event.key == pygame.K_a:
                    if pressed_keys[pygame.K_d]:
                        self.rocket.set_theta(-1.0)  # Angle Booster right
                    else:
                        self.rocket.set_theta(0.0)  # Angle Booster center
                elif event.key == pygame.K_d:
                    if pressed_keys[pygame.K_a]:
                        self.rocket.set_theta(1.0)  # Angle Booster left
                    else:
                        self.rocket.set_theta(0.0)  # Angle Booster center
                elif event.key == pygame.K_LEFT:
                    if pressed_keys[pygame.K_RIGHT]:
                        self.rocket.set_f2(1.0)  # Fire left control nozzle
                    else:
                        self.rocket.set_f2(0.0)  # Stop firing control nozzle
                elif event.key == pygame.K_RIGHT:
                    if pressed_keys[pygame.K_LEFT]:
                        self.rocket.set_f2(-1.0)  # Fire right control nozzle
                    else:
                        self.rocket.set_f2(0.0)  # Stop firing control nozzle

    def _update(self, dt, action=None):
        state = self.rocket.update(dt, action)

        done = False

        reward = 0.0 if action == 0 else -1.0  # Reward if booster is on or off

        engine_on_ground = 0.0 <= (state[1] + self.rocket.l1 * np.cos(state[4]))
        nose_on_ground = 0.0 <= (state[1] - self.rocket.l2 * np.cos(state[4]))
        
        out_of_bounds_left = state[0] < -(self.width/2 / physics.pixel_per_meter)
        out_of_bounds_right = state[0] > (self.width/2 / physics.pixel_per_meter)
        out_of_bounds_top = state[1] < -(self.height / physics.pixel_per_meter)

        if out_of_bounds_left or out_of_bounds_right or out_of_bounds_top:
            done = True
            reward += 0.0  # Reward for flying out of bounds
        elif engine_on_ground or nose_on_ground:
            done = True
            # TODO: Add some complex reward function here

        return state, reward, done

    def _render(self):
        ppm = physics.pixel_per_meter

        dirty_rects = []  # Store alle areas that changed in here
        self.display_surf.fill((255, 255, 255))

        (r_x1, r_y1), (r_x2, r_y2), (r_xb, r_yb), (r_xn, r_yn) = self.rocket.render()

        # The coordinate systems origin is at the bottom center of the screen
        x1 = r_x1 + self.width / 2
        y1 = r_y1 + self.height
        x2 = r_x2 + self.width / 2
        y2 = r_y2 + self.height

        xb = r_xb + self.width / 2
        yb = r_yb + self.height
        xn = r_xn + self.width / 2
        yn = r_yn + self.height

        burn_line = pygame.draw.line(self.display_surf, (239, 151, 0), (x1, y1), (xb, yb), 3)
        nozzle_line = pygame.draw.line(self.display_surf, (239, 151, 0), (x2, y2), (xn, yn), 1)

        rocket_circ1 = pygame.draw.circle(self.display_surf, (0, 0, 0), (x1, y1), 3)
        rocket_circ2 = pygame.draw.circle(self.display_surf, (0, 0, 0), (x2, y2), 3)
        rocket_line = pygame.draw.line(self.display_surf, (0, 0, 0), (x1, y1), (x2, y2), 5)

        # Ground and landing pad
        pygame.draw.rect(self.display_surf, (0, 0, 0), (0, self.height - physics.ground_height * ppm, self.width, physics.ground_height * ppm))
        pygame.draw.line(self.display_surf, (239, 151, 0), (self.width/2 - 10 * ppm, self.height - physics.ground_height * ppm), (self.width/2 + 10 * ppm, self.height - physics.ground_height * ppm), 5)

        new_dirty_rects = [rocket_circ1, rocket_circ2, rocket_line, burn_line, nozzle_line]
        dirty_rects = self.last_dirty_rects + new_dirty_rects

        pygame.display.update(dirty_rects)

        self.last_dirty_rects = new_dirty_rects

    def _cleanup(self):
        pygame.quit()

        # Save Net if agent was trained
        if self.agent_play and self.agent_train:
            self.agent.save_net(self.agent_file)

    def play(self):
        if self._init() == False:
            self.running = False
        
        episode = 0

        # Loop over all Episodes:
        while self.running:
            episode += 1
            print(f'Episode {episode} started')

            state, done = self.rocket.reset(), False

            self._init_time()
            # Loop for one Episode:
            while self.running and not done:
                dt = 0.01  # Will always calculate 100 steps per second, even if realtime simulation is slower

                for event in pygame.event.get():
                    self._on_event(event)
                
                if self.agent_play and self.agent_train:
                    action = self.agent.act_e_greedy(state)  # Exploration strategy for training
                elif self.agent_play and not self.agent_train:
                    action = self.agent.act_greedily(state)  # Optimal strategy for evaluation
                elif not self.agent_play:
                    action = None  # No action because human inputs work with pygame events

                next_state, reward, done = self._update(dt, action)
                
                if self.agent_play and self.agent_train:
                    self.agent.experience(state, action, reward, next_state, done)
                    self.agent.train()
                
                state = next_state

                if self.render:
                    self._render()
                    self._delay(dt)  # Wait in render mode after each calculation to ensure simulation is in real time

        self._cleanup()
    
    def _init_time(self):
        self.old_time_ms = pygame.time.get_ticks()

    def _delay(self, dt):
        new_time_ms = pygame.time.get_ticks()
        delta_time_ms = new_time_ms - self.old_time_ms
        self.old_time_ms = new_time_ms

        d_d_time_ms = int(dt * 1000 - delta_time_ms)

        if d_d_time_ms > 0:
            pygame.time.delay(d_d_time_ms)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device {device}')

    game = Game(render=True, agent_play=False, agent_train=True, agent_file='rocket_game.net', device=device)
    game.play()