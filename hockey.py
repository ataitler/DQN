import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

class HockeyEnv(gym.Env):
	metadata = {
		'render.modes' : ['human', 'rgb_array'],
        	'video.frames_per_second' : 30
	}
	
	def __init__(self):
		self.max_speed = 25.
		self.max_force = 20.
		#self.max_x = 122.
		#self.max_y = 61.
		self.max_x = 50.
		self.max_y = 25.
		self.goal_len = 10
		self.dt = .05
		self.e = 1
		#self.puckR = 3.25
		#self.agentR = 4.75
		self.puckR = 3
		self.agentR = 4
		self.viewer = None

		# observation space - [ax ay aVx aVy px py pVx pVy]
		# X-Y 244x122 [cm]
		obs_high = np.array([self.max_x+self.puckR, self.max_y, self.max_speed, self.max_speed, 
			self.max_x+self.puckR, self.max_y, self.max_speed, self.max_speed])
		ac_high = np.array([self.max_force, self.max_force])
		self.observation_space = spaces.Box(low = -obs_high, high = obs_high)
		self.action_space = spaces.Box(low = -ac_high, high = ac_high)

		self._seed()

	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def _step(self, u):
		Done = False
		agentX, agentY, agentVx, agentVy, puckX, puckY, puckVx, puckVy = self.state

		dt = self.dt
		self.last_u = u
		u = np.clip(u, -self.max_force, self.max_force)
		costs = self.dist(agentX, agentY, puckX, puckY)
		
		# advance puck dynamics	
		newpVx = puckVx
		newpX = puckX + newpVx*dt
		newpVy = puckVy
		newpY = puckY + newpVy*dt
		#puck collision with walls
		effectiveY = self.max_y-self.puckR

		if newpY > effectiveY:					# right wall
			newpVy = -self.e*newpVy
			newpVx = self.e*newpVx
			newpY = effectiveY - (newpY - effectiveY)
		elif newpY < -effectiveY:				# left wall
			newpVy = -self.e*newpVy
			newpVx = self.e*newpVx
			newpY = -effectiveY + (-effectiveY-newpY)
		
		newaVx = agentVx + dt*u[0]
		newaVy = agentVy + dt*u[1]
		newaX = agentX + newaVx*dt
		newaY = agentY + newaVy*dt
		
		### handle terminal states
		# puck hit the agent's wall
		if newpX < -self.max_x or newpX > self.max_x:
			Done = True
			costs = 100000

		# agent hit on of the walls
		#if newaX > (self.max_x-self.agentR) or newaX < (-self.max_x+self.agentR):
		if newaX > (-self.agentR) or newaX < (-self.max_x+self.agentR):
			Done = True
			costs = 100000
		elif newaY > (self.max_y-self.agentR) or newaY < (-self.max_y+self.agentR):
			Done = True
			costs = 100000

		#detect puck agent collision
		if self.dist(newaX, newaY, newpX, newpY) < self.puckR+self.agentR:
			# collision detected
			#newpVx,newpVy = self.calcVelocities(newaX,newaY,newaVx,newaVy,newpX,newpY,newpVx,newpVy)
			#newaVx = 0
			#newaVy = 0
			#newpX = puckX
			#newpuckY = puckY
			Done = True
			costs = -10000

		#clip velocities
		newaVy = np.clip(newaVy, -self.max_speed, self.max_speed)
		newaVx = np.clip(newaVx, -self.max_speed, self.max_speed)
		newpVy = np.clip(newpVy, -self.max_speed, self.max_speed)
		newpVx = np.clip(newpVx, -self.max_speed, self.max_speed)

		#return new state
		self.state = np.array([newaX,newaY,newaVx,newaVy,newpX,newpY,newpVx,newpVy])
		return self._get_obs(), -costs, Done, {}


	def _reset(self):
		#                ax    ay   aVx aVy px  py   pVx pVy
		# standing puck in the agent's side
		low = np.array([-self.max_x + self.agentR + 5.0, -self.max_y + self.agentR + 5.0, 0, 0, -self.puckR, -self.max_y + self.puckR + 5.0, 0, 0])
		high = np.array([-self.max_x + self.agentR + 15., self.max_y - self.agentR - 5.0, 0, 0, -self.max_x/3.0*2.0, self.max_y - self.puckR - 5.0, 0, 0])
		self.state = self.np_random.uniform(low = low, high = high)
		self.last_u = None
		return self._get_obs()

	def _get_obs(self):
		ax, ay, aVx, aVy, px, py, pVx, pVy = self.state
		return np.array([ax, ay, aVx, aVy, px, py, pVx, pVy])

	def _render(self, mode='human', close=False):
		if close:
			if self.viewer is not None:
				self.viewer.close()
				self.viewer = None
			return

		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(700,700)
			self.viewer.set_bounds(-80, 80, -80, 80)
			
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
		
	def dist(self, X1, Y1, X2, Y2):
		return np.sqrt((X1-X2)**2 + (Y1-Y2)**2)

	def calcVelocities(self, aX,aY,aVx,aVy,pX,pY,pVx,pVy):
		colX = np.abs(pX-aX)/2 + np.min([aX,pX])
		colY = np.abs(pY-aY)/2 + np.min([aY,pY])
		theta = np.arctan2(pX-aX, pY-aY)
		colpVx = np.cos(theta)*pVx + np.sin(theta)*pVy
		colpVy = -np.sin(theta)*pVx + np.cos(theta)*pVy
		colaVx = np.cos(theta)*aVx + np.sin(theta)*aVy
		colaVy = -np.sin(theta)*aVx + np.cos(theta)*aVy

		newpVxR = colpVx
		newpVyR = 1*colaVy - colaVy
		
		newpVx = np.cos(theta)*newpVxR - np.sin(theta)*newpVyR
		newpVy = np.sin(theta)*newpVxR + np.cos(theta)*newpVyR

		return newpVx, newpVy

