import pygame
import numpy as np
import physics

class Rocket:
    def __init__(self):
        self.l1 = 1.5  # Distance between center of mass and engine in m
        self.l2 = 3.5  # Distance between center of mass and nose in m
        self.m = 1.0E3  # Overall mass in kg
        self.J = 5.25E3  # Moment of Inertia in kg * m^2

        self.reset()
    
    def reset(self):
        # Coordinates are relative to landing pad in the bottom middle of the screen, y is pointing down
        self.x = (np.random.rand() * 2.0 - 1.0) * 20.0
        self.y = -60.0
        self.phi = -self.x / 100.0 + (np.random.rand() * 2.0 - 1.0) * 0.25

        self.x_v = 0.0
        self.y_v = 30.0
        self.phi_v = 0.0
        
        self.f1 = 0.0
        self.f2 = 0.0
        self.theta = 0.0  # Angle of thrust

        return np.array([self.x, self.y, self.x_v, self.y_v, self.phi, self.phi_v], dtype=np.float32)
    
    def init(self):
        pass
    
    def update(self, dt, action=None):
        if not action == None:
            self.control(action)

        x_a = self._calc_x_a()
        self.x_v += x_a * dt
        self.x += self.x_v * dt
        
        y_a = self._calc_y_a()
        self.y_v += y_a * dt
        self.y += self.y_v * dt

        phi_a = self._calc_phi_a()
        self.phi_v += phi_a * dt
        self.phi += self.phi_v * dt

        if self.phi > np.pi:
            self.phi -= 2.0 * np.pi
        elif self.phi <= -np.pi:
            self.phi += 2.0 * np.pi
        
        # Return state
        return np.array([self.x, self.y, self.x_v, self.y_v, self.phi, self.phi_v], dtype=np.float32)

    def control(self, action):
        if action == 0:
            self.set_f1(0.0)  # Turn off Booster
        else:
            self.set_f1(1.0)  # Turn on Booster
        
        if action == 0 or action == 1:
            self.set_theta(0.0)  # Angle Booster center
        elif action == 2:
            self.set_theta(1.0)  # Angle Booster left
        else:
            self.set_theta(-1.0)  # Angle Booster right

    def set_f1(self, f):
        self.f1 = physics.main_engine_force * f
    
    def set_f2(self, f):
        self.f2 = physics.control_engine_force * f

    def set_theta(self, t):
        self.theta = physics.main_engine_angle * t
    
    def render(self):
        x1 = self.x - self.l1 * np.sin(self.phi)
        y1 = self.y + self.l1 * np.cos(self.phi) - physics.ground_height

        x1_px = x1 * physics.pixel_per_meter
        y1_px = y1 * physics.pixel_per_meter

        x2 = self.x + self.l2 * np.sin(self.phi)
        y2 = self.y - self.l2 * np.cos(self.phi) - physics.ground_height

        x2_px = x2 * physics.pixel_per_meter
        y2_px = y2 * physics.pixel_per_meter

        # Booster flame endpoint
        xb = x1 - 4.0 * np.sin(self.phi + self.theta) * self.f1 / physics.main_engine_force
        yb = y1 + 4.0 * np.cos(self.phi + self.theta) * self.f1 / physics.main_engine_force

        xb_px = xb * physics.pixel_per_meter
        yb_px = yb * physics.pixel_per_meter

        # Nozzle flame endpoint
        xn = x2 - 1.0 * np.cos(self.phi) * self.f2 / physics.control_engine_force
        yn = y2 - 1.0 * np.sin(self.phi) * self.f2 / physics.control_engine_force

        xn_px = xn * physics.pixel_per_meter
        yn_px = yn * physics.pixel_per_meter

        return (x1_px, y1_px), (x2_px, y2_px), (xb_px, yb_px), (xn_px, yn_px)

    def _calc_x_a(self):
        return np.sin(self.phi + self.theta) * self.f1 / self.m

    def _calc_y_a(self):
        return physics.g - np.cos(self.phi + self.theta) * self.f1 / self.m

    def _calc_phi_a(self):
        return (self.f2 * self.l2 - np.sin(self.theta) * self.f1 * self.l1) / self.J