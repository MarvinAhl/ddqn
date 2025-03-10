import pygame
import physics
from rocket_hover import Rocket

from dddqn import DDDQN
import torch
import numpy as np
import matplotlib.pyplot as plt

class Game:
    def __init__(self, render=True, agent_play=True, agent_train=True, agent_file='rocket_game_hover_adv', agent_load_file=None, save_episodes=100,
                 step_limit=2000, device='cpu'):
        self.running = True
        self.display_surf = None
        self.size = (self.width, self.height) = (1280, 720)

        self.rocket = Rocket()
        self.last_dirty_rects = []

        self.agent_play = agent_play
        self.agent_train = agent_train
        self.agent_file = agent_file
        self.agent_load_file = agent_load_file

        if agent_play:
            sampling_period = 0.01
            lookahead_horizon = 5.0
            gamma = np.exp(-sampling_period/lookahead_horizon)  # Calculate discount factor
            self.agent = DDDQN(6, 6, hidden_layers=(1000, 2000, 2000, 2000, 1000), gamma=gamma, learning_rate_start=0.0005, learning_rate_decay_steps=100000, learning_rate_min=0.0003,
                               weight_decay=0.001, epsilon_start=1.0, epsilon_decay_steps=20000, epsilon_min=0.15, temp_start=10, temp_decay_steps=20000, temp_min=0.1,
                               buffer_size_min=200, buffer_size_max=50000, batch_size=50, replays=1, tau=0.01, alpha=0.6, beta=0.1, beta_increase_steps=20000,
                               device=device)
            if not agent_train:
                self.agent.load_net(agent_file + '.net')
            elif agent_train and not agent_load_file == None:
                self.agent.load_net(agent_load_file + '.net')  # Only if should start from an existing network
        else:
            self.agent = None
        
        self.x_target = 0.0
        self.y_target = -30.0

        self.last_shaping = None  # For potential based reward shaping
        
        self.save_episodes = save_episodes
        self.step_limit = step_limit

        self.render = render

        # For measuring frame time
        self.last_time_ms = 0

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

        # Oberservation is like state but only with relative positions
        obsv = state.copy()
        obsv[0] -= self.x_target
        obsv[1] -= self.y_target

        shaping = self._calc_shaping(obsv)
        reward = shaping - self.last_shaping
        self.last_shaping = shaping

        dist = np.sqrt(obsv[0]**2 + obsv[1]**2)
        done = dist > 28.0

        return obsv, reward, done

    def _calc_shaping(self, state):
        # Square potentials scaled to have about the same impact
        position = -0.06 * (state[0]**2 + state[1]**2)
        velocity = -0.03 * (state[2]**2 + state[3]**2)
        angle = -5.0 * state[4]**2
        ang_vel = -12.0 * state[5]**2

        # Scaled by importance
        return 2.0 * position + velocity + 0.5 * angle + 0.5 * ang_vel

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

    def play(self):
        if self._init() == False:
            self.running = False

        episode_rewards = []
        steps = []
        greedy_rates = []
        final_obsvs = []

        episode = 0

        # Loop over all Episodes:
        while self.running:
            episode += 1
            print(f'Episode {episode} started')

            state, done = self.rocket.reset(), False

            obsv = state.copy()
            obsv[0] -= self.x_target
            obsv[1] -= self.y_target

            self.last_shaping = self._calc_shaping(obsv)  # Calculate initial potential

            episode_reward = 0.0
            step_count = 0
            greedy_count = 0

            self._init_time()
            # Loop for one Episode:
            while self.running and not done:
                dt = 0.01  # Will always calculate 100 steps per second, even if realtime simulation is slower

                for event in pygame.event.get():
                    self._on_event(event)
                
                if self.agent_play and self.agent_train:
                    action, is_greedy = self.agent.act_e_greedy(obsv)  # Exploration strategy for training
                elif self.agent_play and not self.agent_train:
                    action, is_greedy = self.agent.act_greedily(obsv)  # Optimal strategy for evaluation
                elif not self.agent_play:
                    action, is_greedy = None, True  # No action because human inputs work with pygame events

                next_obsv, reward, done = self._update(dt, action)
                
                if self.agent_play and self.agent_train:
                    self.agent.experience(obsv, action, reward, next_obsv, done)
                    self.agent.train()

                if self.render:
                    self._render()
                    self._delay(dt)  # Wait in render mode after each calculation to ensure simulation is in real time
                
                obsv = next_obsv

                episode_reward += reward
                step_count += 1
                greedy_count += is_greedy

                # Stop Episode if time limit is reached
                if step_count >= self.step_limit and self.agent_play:
                    done = True
            
            if self.agent_play:
                print(f'Reward: {episode_reward}')
            else:
                print(f'x: {obsv[0]}')
                print(f'y: {obsv[1]}')
                print(f'x_v: {obsv[2]}')
                print(f'y_v: {obsv[3]}')
                print(f'phi: {obsv[4]}')
                print(f'phi_v: {obsv[5]}')
                print(f'Reward: {episode_reward}')
                print('')

            episode_rewards.append(episode_reward)
            overall_steps = step_count if len(steps) < 1 else steps[-1] + step_count
            steps.append(overall_steps)
            greedy_rates.append(greedy_count / step_count)
            final_obsvs.append(obsv)

            if (episode % self.save_episodes == 0 or not self.running) and self.agent_play:
                # Save Net if agent was trained
                if self.agent_train:
                    self.agent.save_net(self.agent_file + f'_e{episode}' + '.net')

                # Plot results
                plt.clf()  # To prevent overlapping of old plots

                figure, axis = plt.subplots(3, 1)
                figure.suptitle('Training Stats')

                axis[0].plot(np.arange(episode) + 1, episode_rewards, 'k-')
                axis[0].grid(True)
                axis[0].set_ylabel('Reward')

                axis[1].plot(np.arange(episode) + 1, steps, 'k-')
                axis[1].grid(True)
                axis[1].set_ylabel('Steps')

                axis[2].plot(np.arange(episode) + 1, greedy_rates, 'k-')
                axis[2].grid(True)
                axis[2].set_ylabel('Greedy Rate')
                axis[2].set_xlabel('Episodes')

                prefix = 'train' if self.agent_train else 'eval'
                plt.savefig(prefix + f'_stats_e{episode}.png')

                plt.close(figure)

                # Plot final obsvs
                plt.clf()  # To prevent overlapping of old plots

                figure, axis = plt.subplots(3, 2)
                figure.suptitle('Final Observations')

                axis[0, 0].plot(np.arange(episode) + 1, [obsv[0] for obsv in final_obsvs], 'k.')
                axis[0, 0].grid(True)
                axis[0, 0].set_ylabel('x/y Position')

                axis[0, 1].plot(np.arange(episode) + 1, [obsv[1] for obsv in final_obsvs], 'k.')
                axis[0, 1].grid(True)

                axis[1, 0].plot(np.arange(episode) + 1, [obsv[2] for obsv in final_obsvs], 'k.')
                axis[1, 0].grid(True)
                axis[1, 0].set_ylabel('x/y Velocity')

                axis[1, 1].plot(np.arange(episode) + 1, [obsv[3] for obsv in final_obsvs], 'k.')
                axis[1, 1].grid(True)

                axis[2, 0].plot(np.arange(episode) + 1, [obsv[4] for obsv in final_obsvs], 'k.')
                axis[2, 0].grid(True)
                axis[2, 0].set_ylabel('phi Angle/Velocity')
                axis[2, 0].set_xlabel('Episodes')

                axis[2, 1].plot(np.arange(episode) + 1, [obsv[5] for obsv in final_obsvs], 'k.')
                axis[2, 1].grid(True)
                axis[2, 1].set_xlabel('Episodes')

                plt.savefig(prefix + f'_final_obsvs_e{episode}.png')

                plt.close(figure)

                print(f'Episode {episode} saved')

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
    
    def _gauss_reward(self, a, b, c, x):
        """
        Calculate unnormalized Gauss Curve with maximum height of a at x=0 that
        decays to c*a at x=b. Return value at point x. Is used for the reward function.
        """
        temp = np.sqrt(-np.log(c**2))
        alpha = np.sqrt(2*np.pi) * a * b / temp
        sigma = b / temp

        return alpha / sigma / np.sqrt(2*np.pi) * np.exp(-x**2 / 2 / sigma**2)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device {device}')

    game = Game(render=True, agent_play=True, agent_train=True, agent_file='rocket_game_hover', save_episodes=100, step_limit=2000, device=device)
    game.play()