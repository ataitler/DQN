import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

class HockeyEnvLeft(gym.Env):
	metadata = {
		'render.modes' : ['human', 'rgb_array'],
        	'video.frames_per_second' : 30
	}
	
	def __init__(self):
		self.normalization_factor = 1.
		# full settings
		#self.max_x = 100.
		#self.max_y = 50.
		#self.max_speed = 100.
		#self.max_force = 50.
		#self.goal_len = 40
		#self.puckR = 3.2
		#self.agentR = 4.5

		# reduced settings
		self.max_speed = 10.
		self.max_force = 10.
		self.max_x = 20.
		self.max_y = 10.
		self.goal_len = 4
		self.puckR = 1
		self.agentR = 2
		self.dis = 10

		self.dt = .05
		self.e = 1
		self.viewer = None
		self.totalCost = 0
		self.max_steps = 300

		self.worst_cost = self.dist(2*self.max_x, 2*self.max_y, 0, 0)**2

		# observation space - [ax ay aVx aVy px py pVx pVy]
		# X-Y 244x122 [cm]
		obs_high = np.array([self.max_x+self.agentR, self.max_y+self.agentR, self.max_speed+1, self.max_speed+1, 
			             self.max_x+self.puckR, self.max_y+self.puckR, self.max_speed+1, self.max_speed+1])
		ac_high = np.array([self.max_force, self.max_force])
		self.observation_space = spaces.Box(low = -obs_high, high = obs_high)
		self.action_space = spaces.Box(low = -ac_high, high = ac_high)

		self._seed()

	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def _step(self, u):
		Done = False
		self.step_number = self.step_number+1
		agentX, agentY, agentVx, agentVy, puckX, puckY, puckVx, puckVy = self.state

		dt = self.dt
		u = np.clip(u, -self.max_force, self.max_force)
		#costs = self.dist(agentX, agentY, puckX, puckY)**2
		
		# advance puck's dynamics	
		newpVx = np.clip(puckVx, -self.max_speed, self.max_speed)
		newpX = round(puckX + newpVx*dt,self.dis)
		newpVy = np.clip(puckVy, -self.max_speed, self.max_speed)
		newpY = round(puckY + newpVy*dt,self.dis)
	
		# advance agent's dynamics	
		newaVx = np.clip(agentVx + u[0]*dt, -self.max_speed, self.max_speed)
		newaVy = np.clip(agentVy + u[1]*dt, -self.max_speed, self.max_speed)
		#newaX = agentX + newaVx*dt
		#newaY = agentY + newaVy*dt
		newaX = round(agentX + newaVx*dt,self.dis)
		newaY = round(agentY + newaVy*dt,self.dis)

		#costs = self.dist(newaX, newaY, newpX, newpY)**2
		costs = 0

		# wall hit
		if (newaX > -self.agentR) or (newaX < self.agentR-self.max_x) or (newaY < -self.max_y+self.agentR) or (newaY > self.max_y-self.agentR):
			Done = True
			#costs = 100

		# puck hit
		elif self.dist(newaX, newaY, newpX, newpY) < self.puckR+self.agentR:
			Done = True
			# calc post collision properties
			newpVx, newpVy = self.calcVelocities(newaX,newaY,newaVx,newaVy,newpX,newpY,newpVx,newpVy)
			newpVx = np.clip(newpVx, -self.max_speed, self.max_speed)
			newpVy = np.clip(newpVy, -self.max_speed, self.max_speed)
			#if newpVx < 0:
			#	costs = 50
			#else:
			speed, goal_point = self.calcTrajectoryProperties(newpX,newpY,newpVx,newpVy,0)
			costs = -self.calcReward(goal_point, speed)

		costs = costs/self.normalization_factor
		self.state = np.array([newaX,newaY,newaVx,newaVy,newpX,newpY,newpVx,newpVy])
		return self._get_obs(), -costs, Done, {}


	def _reset(self):
		#                ax    ay   aVx aVy px  py   pVx pVy
		# standing puck in the agent's side
		# agent is random puck is random 
		#low = np.array([-self.max_x + 1. + self.agentR,  -self.max_y + self.agentR + 1., -self.max_speed/3., -self.max_speed/2., round(-self.puckR,1), -self.max_y + self.puckR + 5.0, 0, 0])
		#high = np.array([-self.max_x + 1. + self.agentR, self.max_y - self.agentR - 1., self.max_speed/3., self.max_speed/2., round(-self.max_x/3.5*2.0,1), self.max_y - self.puckR - 5.0, 0, 0])
		
		# agent is random (in position) puck is random 
		#low = np.array([-self.max_x + 1. + self.agentR,  -self.max_y + self.agentR + 1., 0, 0, round(-self.puckR,1), -self.max_y + self.puckR + 5.0, 0, 0])
		#high = np.array([-self.max_x + 1. + self.agentR, self.max_y - self.agentR - 1., 0, 0, round(-self.max_x/3.5*2.0,1), self.max_y - self.puckR - 5.0, 0, 0])

		# agent is fixed puck is random
		#low = np.array([-self.max_x + 1. + self.agentR, 0.0, 0, 0, round(-self.puckR,1), -self.max_y + self.puckR + 5.0, 0, 0])
		#high = np.array([-self.max_x + 1. + self.agentR, 0.0, 0, 0, round(-self.max_x/3.5*2.0,1), self.max_y - self.puckR - 5.0, 0, 0])

		# both agent and puck are fixed
		#low = np.array([-self.max_x + 1. + self.agentR, 0.0, 0, 0, -5.0, -5.0, 0, 0])
		#high = np.array([-self.max_x + 1. + self.agentR, 0.0, 0, 0, -5.0, -5.0, 0, 0])
		
		# both are fixed in the left side of the table
		low = np.array([-self.max_x + 1. + self.agentR, 0.0, 0, 0, -5.0, -self.max_y + self.puckR + 5, 0, 0])
		high = np.array([-self.max_x + 1. + self.agentR, 0.0, 0, 0, -5.0, -self.max_y + self.puckR + 5, 0, 0])

		self.state = np.round(self.np_random.uniform(low = low, high = high),1)
		self.totalCost = 0
		self.step_number = 0
		return self._get_obs()

	def _get_obs(self):
		ax, ay, aVx, aVy, px, py, pVx, pVy = self.state
		return np.array([ax, ay, aVx, aVy, px, py, pVx, pVy])

		
	def dist(self, X1, Y1, X2, Y2):
		return np.sqrt((X1-X2)**2 + (Y1-Y2)**2)

	def calcVelocities(self, ax,ay,aVx,aVy,pX,pY,pVx,Pvy):
		ex = pX - ax
		ey = pY - ay
		norma = np.sqrt(ex**2 + ey**2)
		ex = ex/norma
		ey = ey/norma

		#velocity transfered to the puck
		vel_norm = self.e * np.sqrt(aVx**2 + aVy**2)
		Vx = vel_norm * ex
		Vy = vel_norm * ey

		return Vx, Vy

	def calcTrajectoryProperties(self,px,py,pVx,pVy,target_point):
		# distance to goal line
		d_goal_line = self.max_x - px
		# time to goal line
		if pVx == 0:
			#return np.nan, np.nan, np.nan, np.nan
			return np.nan, np.nan
		elif pVx > 0:
			t_goal_line = d_goal_line / pVx
		else:
			return np.nan, np.nan

		# cross point on goal line
		d_y = t_goal_line * pVy + py

		# compute the projection of the velocity on the direction of the target point
		ex = self.max_x - px
		ey = target_point - py
		if ey == 0:
			theta = np.arctan(np.infty)
		else:
			theta = np.arctan(np.absolute(ex/ey))
		V = pVx*np.sin(theta) + np.maximum(0,pVy*np.cos(theta)*np.sign(ey))

		return V, d_y

		#V_norm = np.sqrt(pVy**2 + pVx**2)
		
		#return V_norm, d_y
	
		# folding to regualr board
		#aux_y = np.floor(np.absolute(d_y) / self.max_y)
		#num_of_collisions = 0
		#if aux_y == 0:					# determine the number of wall hits
		#	num_of_collisions = 0
		#	first_wall_hit = 0
		#elif aux_y % 2 == 0:
		#	num_of_collisions = aux_y/2
		#else:
		#	num_of_collisions = (aux_y+1)/2
		
		#if num_of_collisions > 0:			# determine the first wall hit
		#	if pVy < 0:
		#		first_wall_hit = -1
		#	else:
		#		first_wall_hit = 1
		#else: 
		#	first_wall_hit = 0
		#if d_y < 0: 					# left side
		#	d_y_r = d_y + 2*self.max_y*num_of_collisions
		#else:
		#	d_y_r = d_y - 2*self.max_y*num_of_collisions
		#if num_of_collisions % 2 == 1:			# flip point
		#	d_y_r = -d_y_r
		
		#Vy_at_goal = pVy * np.power(self.e, num_of_collisions)
		#V_norm = np.sqrt( Vy_at_goal**2 + pVx**2 )

		#return num_of_collisions, first_wall_hit, d_y_r, V_norm
		
			
	def calcReward(self, goal_point, speed):
		# problem with calculations - maybe collision speed is zero (rounded to zero)
		if np.isnan(speed):
			return 0

		# middle attack reward calculator!!!
		#eps = 0.01
		#W = np.array([1,1,2])
		# wall hits, speed^2, distance from goal^4
		#features = np.sqrt(np.array([1/(num_of_collisions+eps)**2, speed**4, np.power(np.absolute(self.max_y) - np.absolute(goal_point),4)]))
		#features = np.array([speed**2, 100*np.exp(-np.absolute(goal_point))])
		
		x = np.absolute(goal_point)
		b = 50
		W = np.array([1,1])
		features = np.array([np.sign(speed)*speed**2, 100* (self._h(x)-self._h(x-1) + np.exp(-2*(x-1))*self._h(x-1))])
		temp_reward = np.dot(W,features) + b
		return temp_reward

	def _h(self,x):
		if x >= 0:
			return 1
		else:
			return 0

	def sizes(self):
		return self.max_x*2, self.max_y*2, self.agentR, self.puckR
	
	def limits(self):
		return self.max_x, self.max_y, self.max_force, self.max_speed

	def set_max_steps(self, steps):
		self.max_steps = steps

	def _render(self, mode='human', close=False):
		if close:
			if self.viewer is not None:
				self.viewer.close()
				self.viewer = None
			return

		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(700,700)
			self.viewer.set_bounds(-(self.max_x+5), (self.max_x+5), -(self.max_x+5), (self.max_x+5))
			
			# left wall
			left_wall = rendering.make_capsule(self.max_x*2,1)
			left_wall.set_color(0.5,0.3,0.3)
			left_wall_transform = rendering.Transform()
			left_wall.add_attr(left_wall_transform)
			self.viewer.add_geom(left_wall)
			left_wall_transform.set_rotation(np.pi/2)
			left_wall_transform.set_translation(-self.max_y,-self.max_x)
			
			# right wall
			right_wall = rendering.make_capsule(self.max_x*2,1)
			right_wall.set_color(0.5,0.3,0.3)
			right_wall_transform = rendering.Transform()
			right_wall.add_attr(right_wall_transform)
			self.viewer.add_geom(right_wall)
			right_wall_transform.set_rotation(np.pi/2)
			right_wall_transform.set_translation(self.max_y,-self.max_x)
			
			# upper wall (puck's side)
			upper_wall = rendering.make_capsule(self.max_y*2,1)
			upper_wall.set_color(0.5,0.3,0.3)
			upper_wall_transform = rendering.Transform()
			upper_wall.add_attr(upper_wall_transform)
			self.viewer.add_geom(upper_wall)
			upper_wall_transform.set_translation(-self.max_y,-self.max_x)

			# upper goal
			upper_goal = rendering.make_capsule(self.goal_len,1)
			upper_goal.set_color(1.,1.,1.)
			upper_goal_transform = rendering.Transform()
			upper_goal.add_attr(upper_goal_transform)
			self.viewer.add_geom(upper_goal)
			upper_goal_transform.set_translation(-self.goal_len/2., -self.max_x)

			# lower wall (agent's side)
			lower_wall = rendering.make_capsule(self.max_y*2,1)
			lower_wall.set_color(0.5,0.3,0.3)
			lower_wall_transform = rendering.Transform()
			lower_wall.add_attr(lower_wall_transform)
			self.viewer.add_geom(lower_wall)
			lower_wall_transform.set_translation(-self.max_y,self.max_x)

			# lower goal
			lower_goal = rendering.make_capsule(self.goal_len,1)
			lower_goal.set_color(1.,1.,1.)
			lower_goal_transform = rendering.Transform()
			lower_goal.add_attr(lower_goal_transform)
			self.viewer.add_geom(lower_goal)
			lower_goal_transform.set_translation(-self.goal_len/2., self.max_x)
			
			# middle line
			middle_line = rendering.make_capsule(self.max_y*2,1)
			middle_line.set_color(0.1,0.3,0.3)
			middle_line_transform = rendering.Transform()
			middle_line.add_attr(middle_line_transform)
			self.viewer.add_geom(middle_line)
			middle_line_transform.set_translation(-self.max_y,0.)

			middle_circle_big = rendering.make_circle(self.puckR+0.2)
			middle_circle_big.set_color(0.1,0.3,0.3)
			middle_circle_big_transform = rendering.Transform()
			middle_circle_big.add_attr(middle_circle_big_transform)
			self.viewer.add_geom(middle_circle_big)
			middle_circle_big_transform.set_translation(0.,0.)
			
			middle_circle_small = rendering.make_circle(self.puckR-0.5)
			middle_circle_small.set_color(1.,1.,1.)
			middle_circle_small_transform = rendering.Transform()
			middle_circle_small.add_attr(middle_circle_small_transform)
			self.viewer.add_geom(middle_circle_small)
			middle_circle_small_transform.set_translation(0.,0.)
			
			# objects
			puck = rendering.make_circle(self.puckR)
			puck.set_color=(.8,.3,.3)
			self.puck_transform = rendering.Transform()
			puck.add_attr(self.puck_transform)
			self.viewer.add_geom(puck)
			agent = rendering.make_circle(self.agentR)
			agent.set_color(.6,.3,.3)
			self.agent_transform = rendering.Transform()
			agent.add_attr(self.agent_transform)
			self.viewer.add_geom(agent)

		self.puck_transform.set_translation(self.state[5],self.state[4])
		self.agent_transform.set_translation(self.state[1],self.state[0])
		
		return self.viewer.render(return_rgb_array = mode=='rgb_array')


