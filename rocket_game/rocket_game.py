import pygame
import physics
from rocket import Rocket

from dddqn import DDDQN
import torch
import numpy as np
import matplotlib.pyplot as plt

class Game:
    def __init__(self, render=True, agent_play=True, agent_train=True, agent_file='rocket_game', save_episodes=100, step_limit=2000, device='cpu'):
        self.running = True
        self.display_surf = None
        self.size = (self.width, self.height) = (1280, 720)

        self.rocket = Rocket()
        self.last_dirty_rects = []

        self.agent_play = agent_play
        self.agent_train = agent_train
        self.agent_file = agent_file

        if agent_play:
            self.agent = DDDQN(6, 4, hidden_layers=(1000, 2000, 2000, 2000, 1000), gamma=0.99, learning_rate_start=0.0005, learning_rate_decay_steps=50000, learning_rate_min=0.0003,
                               weight_decay=0.0005, epsilon_start=1.0, epsilon_decay_steps=20000, epsilon_min=0.15, temp_start=10, temp_decay_steps=20000, temp_min=0.1,
                               buffer_size_min=200, buffer_size_max=50000, batch_size=50, replays=1, tau=0.01, alpha=0.6, beta=0.1, beta_increase_steps=20000,
                               device=device)
            if not agent_train:
                self.agent.load_net(agent_file + '.net')
        else:
            self.agent = None
        
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

        done = False

        reward = 0.0 if action == 0 else -0.01  # Reward if booster is off or on

        engine_on_ground = 0.0 <= (state[1] + self.rocket.l1 * np.cos(state[4]))
        nose_on_ground = 0.0 <= (state[1] - self.rocket.l2 * np.cos(state[4]))
        
        out_of_bounds_left = state[0] < -(self.width/2 / physics.pixel_per_meter)
        out_of_bounds_right = state[0] > (self.width/2 / physics.pixel_per_meter)
        out_of_bounds_top = state[1] < -(self.height / physics.pixel_per_meter)

        if out_of_bounds_left or out_of_bounds_right or out_of_bounds_top:
            done = True
            reward += -50.0  # Reward for flying out of bounds
        elif engine_on_ground:
            done = True
            # Reward for x-Position
            reward += self._gauss_reward(10.0, 30.0, 0.15, state[0])  # Low, flat curve to give direction
            reward += self._gauss_reward(10.0, 2.0, 0.15, state[0])  # High, sharp peak to really reward perfect behavior
            
            # Guiding Rewards here
            # Slight reward for y-Velocity
            y_v_fac = self._gauss_reward(1.0, 25.0, 0.15, state[3])
            reward += 50.0 * y_v_fac
            # Add additional reward for good x-Velocity if y-Velocity is good
            x_v_fac = self._gauss_reward(1.0, 10.0, 0.15, state[2])
            reward += 20.0 * x_v_fac * y_v_fac
            # Same for angle
            phi_fac = self._gauss_reward(1.0, 0.5, 0.15, state[4])
            reward += 20.0 * phi_fac * x_v_fac * y_v_fac
            # And Angular Momentum
            phi_v_fac = self._gauss_reward(1.0, 0.5, 0.15, state[5])
            reward += 20.0 * phi_v_fac * phi_fac * x_v_fac * y_v_fac
            
            x_v_good = state[2] < 5.0 and state[2] > -5.0
            y_v_good = state[3] < 10.0 and state[3] > -10.0
            phi_good = state[4] < 0.2 and state[4] > -0.2
            phi_v_good = state[5] < 0.4 and state[5] > -0.4

            rocket_landed = x_v_good and y_v_good and phi_good and phi_v_good
            if rocket_landed:
                reward += 100.0  # Reward for landing

                # Rewards for being on point
                reward += self._gauss_reward(20.0, 3.0, 0.15, state[2])
                reward += self._gauss_reward(20.0, 1.0, 0.15, state[3])
                reward += self._gauss_reward(20.0, 0.1, 0.15, state[4])
                reward += self._gauss_reward(20.0, 0.2, 0.15, state[5])
            else:
                reward += -50.0
        elif nose_on_ground:
            done = True
            reward += -100.0

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

    def play(self):
        if self._init() == False:
            self.running = False

        episode_rewards = []
        steps = []
        greedy_rates = []
        final_states = []

        episode = 0

        # Loop over all Episodes:
        while self.running:
            episode += 1
            print(f'Episode {episode} started')

            state, done = self.rocket.reset(), False

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
                    action, is_greedy = self.agent.act_e_greedy(state)  # Exploration strategy for training
                elif self.agent_play and not self.agent_train:
                    action, is_greedy = self.agent.act_greedily(state)  # Optimal strategy for evaluation
                elif not self.agent_play:
                    action, is_greedy = None, True  # No action because human inputs work with pygame events

                next_state, reward, done = self._update(dt, action)
                
                if self.agent_play and self.agent_train:
                    self.agent.experience(state, action, reward, next_state, done)
                    self.agent.train()

                if self.render:
                    self._render()
                    self._delay(dt)  # Wait in render mode after each calculation to ensure simulation is in real time
                
                state = next_state

                episode_reward += reward
                step_count += 1
                greedy_count += is_greedy

                # Stop Episode if time limit is reached
                if step_count >= self.step_limit and self.agent_play:
                    done = True
            
            if self.agent_play:
                print(f'Reward: {episode_reward}')
            else:
                print(f'x: {state[0]}')
                print(f'y: {state[1]}')
                print(f'x_v: {state[2]}')
                print(f'y_v: {state[3]}')
                print(f'phi: {state[4]}')
                print(f'phi_v: {state[5]}')
                print(f'Reward: {episode_reward}')
                print('')

            episode_rewards.append(episode_reward)
            overall_steps = step_count if len(steps) < 1 else steps[-1] + step_count
            steps.append(overall_steps)
            greedy_rates.append(greedy_count / step_count)
            final_states.append(state)

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

                plt.savefig(f'train_stats_e{episode}.png')

                plt.close(figure)

                # Plot final states
                plt.clf()  # To prevent overlapping of old plots

                figure, axis = plt.subplots(3, 2)
                figure.suptitle('Final States')

                axis[0, 0].plot(np.arange(episode) + 1, [state[0] for state in final_states], 'k.')
                axis[0, 0].grid(True)
                axis[0, 0].set_ylabel('x/y Position')

                axis[0, 1].plot(np.arange(episode) + 1, [state[1] for state in final_states], 'k.')
                axis[0, 1].grid(True)

                axis[1, 0].plot(np.arange(episode) + 1, [state[2] for state in final_states], 'k.')
                axis[1, 0].grid(True)
                axis[1, 0].set_ylabel('x/y Velocity')

                axis[1, 1].plot(np.arange(episode) + 1, [state[3] for state in final_states], 'k.')
                axis[1, 1].grid(True)

                axis[2, 0].plot(np.arange(episode) + 1, [state[4] for state in final_states], 'k.')
                axis[2, 0].grid(True)
                axis[2, 0].set_ylabel('phi Angle/Velocity')
                axis[2, 0].set_xlabel('Episodes')

                axis[2, 1].plot(np.arange(episode) + 1, [state[5] for state in final_states], 'k.')
                axis[2, 1].grid(True)
                axis[2, 1].set_xlabel('Episodes')

                plt.savefig(f'final_states_e{episode}.png')

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

    game = Game(render=True, agent_play=True, agent_train=True, agent_file='rocket_game', save_episodes=100, step_limit=2000, device=device)
    game.play()