import pygame
import physics
from rocket import Rocket

from dddqn import DDDQN
import numpy as np
import matplotlib.pyplot as plt

class Game:
    def __init__(self):
        self.running = True
        self.display_surf = None
        self.size = (self.width, self.height) = (1280, 720)

        self.rocket = Rocket()
        self.last_dirty_rects = []

        self.agent = DDDQN(6, 4, hidden_layers=(500, 500, 500), gamma=0.99, learning_rate_start=0.0005, learning_rate_decay_steps=50000, learning_rate_min=0.0003,
                           epsilon_start=1.0, epsilon_decay_steps=20000, epsilon_min=0.1, temp_start=10, temp_decay_steps=20000, temp_min=0.1, buffer_size_min=200,
                           buffer_size_max=50000, batch_size=50, replays=1, tau=0.01, alpha=0.6, beta=0.1, beta_increase_steps=20000, device=device)

        # for measuring frame time
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
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.rocket.set_f1(1.0)  # Activate Booster
                pass
            elif event.key == pygame.K_a:
                self.rocket.set_theta(1.0)  # Angle Booster left
                pass
            elif event.key == pygame.K_d:
                self.rocket.set_theta(-1.0)  # Angle Booster right
                pass
            elif event.key == pygame.K_LEFT:
                self.rocket.set_f2(-1.0)  # Fire right control nozzle
                pass
            elif event.key == pygame.K_RIGHT:
                self.rocket.set_f2(1.0)  # Fire left control nozzle
                pass

        elif event.type == pygame.KEYUP:
            pressed_keys = pygame.key.get_pressed()
            if event.key == pygame.K_SPACE:
                self.rocket.set_f1(0.0)  # Booster engine off
                pass
            elif event.key == pygame.K_a:
                if pressed_keys[pygame.K_d]:
                    self.rocket.set_theta(-1.0)  # Angle Booster right
                    pass
                else:
                    self.rocket.set_theta(0.0)  # Angle Booster center
                    pass
            elif event.key == pygame.K_d:
                if pressed_keys[pygame.K_a]:
                    self.rocket.set_theta(1.0)  # Angle Booster left
                    pass
                else:
                    self.rocket.set_theta(0.0)  # Angle Booster center
                    pass
            elif event.key == pygame.K_LEFT:
                if pressed_keys[pygame.K_RIGHT]:
                    self.rocket.set_f2(1.0)  # Fire left control nozzle
                    pass
                else:
                    self.rocket.set_f2(0.0)  # Stop firing control nozzle
                    pass
            elif event.key == pygame.K_RIGHT:
                if pressed_keys[pygame.K_LEFT]:
                    self.rocket.set_f2(-1.0)  # Fire right control nozzle
                    pass
                else:
                    self.rocket.set_f2(0.0)  # Stop firing control nozzle
                    pass

    def _update(self, dt):
        self.rocket.update(dt)

    def _render(self):
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

        burn_line = pygame.draw.line(self.display_surf, (150, 50, 0), (x1, y1), (xb, yb), 3)
        nozzle_line = pygame.draw.line(self.display_surf, (150, 50, 0), (x2, y2), (xn, yn), 1)

        rocket_circ1 = pygame.draw.circle(self.display_surf, (0, 0, 0), (x1, y1), 3)
        rocket_circ2 = pygame.draw.circle(self.display_surf, (0, 0, 0), (x2, y2), 3)
        rocket_line = pygame.draw.line(self.display_surf, (0, 0, 0), (x1, y1), (x2, y2), 5)

        new_dirty_rects = [rocket_circ1, rocket_circ2, rocket_line, burn_line, nozzle_line]
        dirty_rects = self.last_dirty_rects + new_dirty_rects

        pygame.display.update(dirty_rects)

        self.last_dirty_rects = new_dirty_rects

    def _cleanup(self):
        pygame.quit()

    def play(self):
        if self._init() == False:
            self.running = False
        
        self._init_time()
        while (self.running):
            dt = self._calc_delta_time()

            for event in pygame.event.get():
                self._on_event(event)

            # do multiple physics calculations per frame to increase simulation accuracy
            for i in range(self.updates_per_frame):
                self._update(dt / self.updates_per_frame)
            self._render()
        
        self._cleanup()
    
    def _init_time(self):
        self.old_time_ms = pygame.time.get_ticks()

    def _calc_delta_time(self):
        new_time_ms = pygame.time.get_ticks()
        delta_time_ms = new_time_ms - self.old_time_ms
        self.old_time_ms = new_time_ms

        # return delta t in seconds
        return delta_time_ms / 1000


if __name__ == "__main__":
    game = Game()
    game.play()