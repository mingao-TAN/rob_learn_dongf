import gym
import time
import numpy as np
import os
from os.path import join
import subprocess
import math
import numpy as np
from waypoint import GlobalPlanSampler
from gym.spaces import Box
from geometry_msgs.msg import PointStamped
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker
import numpy as np
try:  # make sure to create a fake environment without ros installed
    import rospy
    import rospkg
except ModuleNotFoundError:
    pass

from envs.gazebo_simulation import GazeboSimulation
from geometry_msgs.msg import PoseWithCovarianceStamped, Point, Quaternion, Vector3
xxxxxxx = 0.0
yyyyyyy = 0.0
class JackalGazebo(gym.Env):
    def __init__(
        self,
        world_name="jackal_world.world",
        gui=False,
        init_position=[0, 0, 0],
        goal_position=[4, 0, 0],
        max_step=100,
        time_step= 0.1,
        slack_reward=-1,
        failure_reward=-50,
        success_reward=0,
        collision_reward=0,
        goal_reward=1,
        max_collision=10000,
        verbose=True,
        init_sim=True
    ):
        """Base RL env that initialize jackal simulation in Gazebo
        """
        super().__init__()
        # config
        self.gui = gui
        self.verbose = verbose
        self.init_sim = init_sim
        
        # sim config
        self.world_name = world_name
        self.init_position = init_position
        self.goal_position = goal_position
        self.target_point_x = None 
        self.target_point_y = None 
        self.linear_velocity = None
        self.angular_velocity = None
        
        # env config
        self.time_step = time_step
        self.max_step = max_step
        self.slack_reward = slack_reward
        self.failure_reward = failure_reward
        self.success_reward = success_reward
        self.collision_reward = collision_reward
        self.goal_reward = goal_reward
        self.max_collision = max_collision
        self.world_frame_goal = (
            self.init_position[0] + self.goal_position[0],
            self.init_position[1] + self.goal_position[1],
        )
        rospy.init_node('gym', anonymous=True, log_level=rospy.FATAL)
        rospy.set_param('/use_sim_time', True)
        self.subscriber1 = rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback) 
        self.marker_pub = rospy.Publisher('pool_state_marker', Marker, queue_size=10)    
        self.gazebo_sim = GazeboSimulation(init_position=self.init_position)
        self.subscriber = rospy.Subscriber('/target_points', PointStamped, self.callback) 
        # launch gazebo
        if init_sim:
            rospy.logwarn(">>>>>>>>>>>>>>>>>> Load world: %s <<<<<<<<<<<<<<<<<<" %(world_name))
            rospack = rospkg.RosPack()
            self.BASE_PATH = rospack.get_path('jackal_helper')
            world_name = join(self.BASE_PATH, "worlds/BARN", world_name)
            launch_file = join(self.BASE_PATH, 'launch', 'gazebo_launch.launch')

            self.gazebo_process = subprocess.Popen(['roslaunch', 
                                                    launch_file,
                                                    'world_name:=' + world_name,
                                                    'gui:=' + ("true" if gui else "false"),
                                                    'verbose:=' + ("true" if verbose else "false"),
                                                    ])
            time.sleep(10)  # sleep to wait until the gazebo being created

            # initialize the node for gym env
            rospy.init_node('gym', anonymous=True, log_level=rospy.FATAL)
            rospy.set_param('/use_sim_time', True)

            self.gazebo_sim = GazeboSimulation(init_position=self.init_position)

        # place holders
        self.action_space = None
        self.observation_space = None

        self.step_count = 0
        self.collision_count = 0
        self.collided = 0
        self.start_time = self.current_time = None
        
    def seed(self, seed):
        np.random.seed(seed)
        
    def model_states_callback(self,msg):
     robot_name = 'jackal'  # 替换为你的机器人名称
     if robot_name in msg.name:
        # 找到机器人的索引
        index = msg.name.index(robot_name)
        self.linear_velocity = msg.twist[index].linear
        self.angular_velocity = msg.twist[index].angular
       
      
    def callback(self, msg):
        """处理接收到的目标点消息"""
        self.target_point_x = msg.point.x
        self.target_point_y = msg.point.y  
    
    def reset(self):
        raise NotImplementedError

    def step(self, action):
        """take an action and step the environment
        """
        #action[0] = np.clip(action[0] *2.0,v-self.max_acc[0]*self.control_period,v+self.max_acc[0]*self.control_period
        action[0] = action[0] *2.0
        action[1] = action[1] *1.57
        x1 = time.time()
        self._take_action(action)
        x2 = time.time()
        print("action",action)
        self.step_count += 1
        pos, psi = self._get_pos_psi()
        x3 = time.time()
        self.gazebo_sim.unpause()
        # compute observation
        x4 = time.time()
        obs = self._get_observation(pos, psi, action)
        x5 = time.time()
        #转换成float32？
        obs = obs.astype(np.float32)
        flip = pos.z > 0.1  # robot flip
        goal_pos = np.array([self.world_frame_goal[0] - pos.x, self.world_frame_goal[1] - pos.y])
        # goal_pos = np.array([self.world_frame_goal[0] + pos.y, self.world_frame_goal[1] - pos.x])
        
        success = np.linalg.norm(goal_pos) < 0.4
        
        timeout = self.step_count >= self.max_step
        
        collided = self.gazebo_sim.get_hard_collision() and self.step_count > 1
        self.collision_count += int(collided)
        
        done = flip or success or timeout or self.collision_count >= self.max_collision
        
        # compute reward
        rew = self.slack_reward
        if done and not success:
            rew += self.failure_reward
        if success:
            rew += self.success_reward
        if collided:
            rew += self.collision_reward

        rew += (np.linalg.norm(self.last_goal_pos) - np.linalg.norm(goal_pos)) * self.goal_reward
        self.last_goal_pos = goal_pos
        
        info = dict(
            collision=self.collision_count,
            collided=collided,
            goal_position=goal_pos,
            time=self.current_time - self.start_time,
            success=success,
            world=self.world_name
        )
        x6 = time.time()
        if done:
            bn, nn = self.gazebo_sim.get_bad_vel_num()
        x7 = time.time()
        self.gazebo_sim.pause()
        x8 = time.time()
        print("x1", x1)
        print("x2", x2)
        print("x3", x3)
        print("x4", x4)
        print("x5", x5)
        print("x6", x6)
        print("x7", x7)
        print("x8", x8)
        return obs, rew, done, info

    def _take_action(self, action):
        current_time = rospy.get_time()
        while current_time - self.current_time < self.time_step:
            time.sleep(0.01)
            current_time = rospy.get_time()
        self.current_time = current_time

    def _get_observation(self, pos, psi):
        raise NotImplementedError()

        
    def _get_pos_psi(self):
        pose = self.gazebo_sim.get_model_state().pose
        pos = pose.position
        
        # q1 = pose.orientation.x
        q1 = pose.orientation.x
        q2 = pose.orientation.y
        q3 = pose.orientation.z
        q0 = pose.orientation.w
        psi = np.arctan2(2 * (q0*q3 + q1*q2), (1 - 2*(q2**2+q3**2)))
        assert -np.pi <= psi <= np.pi, psi
        
        return pos, psi

    def close(self):
        # These will make sure all the ros processes being killed
        os.system("killall -9 rosmaster")
        os.system("killall -9 gzclient")
        os.system("killall -9 gzserver")
        os.system("killall -9 roscore")


class JackalGazeboLaser(JackalGazebo):
    def __init__(self, laser_clip=4, **kwargs):
        super().__init__(**kwargs)
        self.laser_clip = laser_clip

        # obs_dim = 36 + 2 + self.action_dim  # 720 dim laser scan + goal position + action taken in this time step 
        obs_dim = 540 +8
        self.observation_space = Box(
            low=0,
            high=self.laser_clip,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def _get_laser_scan(self):
        """Get 720 dim laser scan
        Returns:
            np.ndarray: (720,) array of laser scan 
        """
        laser_scan = self.gazebo_sim.get_laser_scan()
        laser_scan = np.array(laser_scan.ranges)
        # laser_scan[laser_scan > self.laser_clip] = self.laser_clip
        return laser_scan

    # def _get_observation(self, pos, psi, action):
    #     # observation is the 720 dim laser scan + one local goal in angle
    #     laser_scan = self._get_laser_scan()
        
    #     # reshape laser scan
    #     reshaped_scan = laser_scan.reshape(36, -1)
        
    #     # min pooling along the second axis
    #     pooled_scan = np.min(reshaped_scan, axis=1) - 0.12
        
    #     # Take the reciprocal of the result after pooling and deal with the zero value problem at the same time
    #     reciprocal_laser_scan = 1.0 / np.where(pooled_scan <= 0, 5e-3 , pooled_scan) 
        
    #     # clip the reciprocal scans
    #     #reciprocal_laser_scan[reciprocal_laser_scan > self.laser_clip] = self.laser_clip

    #     laser_scan = reciprocal_laser_scan

    #     # # normalize
    #     #laser_scan = (laser_scan - self.laser_clip/2.) / self.laser_clip * 2 # scale to (-1, 1)
        
    #     goal_pos = self.transform_goal(self.world_frame_goal, pos, psi)   # roughly (-1, 1) range
    #     rela_angle = np.arctan2(goal_pos[0],goal_pos[1])
    #     rela_dis = np.sqrt(goal_pos[0]**2+goal_pos[1]**2)
        
    #     obs = [laser_scan, [rela_angle,rela_dis], action]
        
    #     obs = np.concatenate(obs)
    #     return obs
    
    # depend on the 90*6 array obsversion 
    def _get_observation(self, pos, psi, action):
        linear_velocity = self.linear_velocity
        angular_velocity = self.angular_velocity
        linear_x = linear_velocity.x
        angular_z = angular_velocity.z
        action_now = [linear_x,angular_z]
        print("actionxxxx",action)
        length1 = 0.210  +0.05
        length2 =0.210 +0.05
        width = 0.2128 +0.08
        max_action_0 = 2.0
        max_action_1 =  1.57
        max_acc_0 = 3.0
        max_acc_1 =  3.0
        # observation is the 720 dim laser scan + one local goal in angle
        laser_scan = self._get_laser_scan() 
        # print(laser_scan)
        # reshape laser scan
        discretized_ranges = []
        for i, item in enumerate(laser_scan):
             if laser_scan[i] == float ('Inf'):
                 discretized_ranges.append(5.0)
             elif np.isnan(laser_scan[i]):
                 discretized_ranges.append(5.0)
            #  elif laser_scan[i] == 0.0:
            #      discretized_ranges.append(0.00001)             
             else:
                 discretized_ranges.append((laser_scan[i]))
        
        laser_scan = np.array(discretized_ranges)
        # reshaped_scan = laser_scan.reshape(360, -1)
        
        # # min pooling along the second axis
        pooled_scan = laser_scan
        # print(pooled_scan)
        data_length = 720
        # ------------------------ change by mingao： 计算四分之一长度 ------------------------  #
        quarter_length = data_length // 4
        # ------------------------ change by mingao： 重新排列数据，从右侧开始逆时针扫描 ------------------------  #
        data_ranges1 = np.concatenate( (pooled_scan[quarter_length:data_length],pooled_scan[0:quarter_length]) )
        pooled_scan = data_ranges1
        
        pool_state = np.zeros((90,6))
        for i in range(90):
            start_idx = 8*i
            end_inx = start_idx + 8
            min_index = np.argmin(pooled_scan[ start_idx : end_inx])
            angle = (start_idx+ min_index) * np.pi/360.0
            x_0 = np.cos(angle)
            y_0 = np.sin(angle)
            pool_state[i,0]  =  x_0
            pool_state[i,1]  =   y_0
            # dis = np.min(pooled_scan[4*i:(4*i+4)])
            dis = pooled_scan[start_idx+min_index]
            # dis = pooled_scan[i]
            pool_state[i,2] = dis
            pool_state[i,3] = length1
            pool_state[i,4] = length2
            pool_state[i,5] = width   
            x_dis =  pool_state[i, 0] *  pool_state[i, 2] 
            y_dis = pool_state[i, 1] * pool_state[i, 2]  
            diss =  0.065
            min_value3 = math.sqrt ( x_dis*x_dis + (y_dis  +diss) *(y_dis  +diss) )
            pool_state[i, 0]  =  (x_dis  / min_value3)
            pool_state[i, 1]  =  ((y_dis +diss) / min_value3)
            pool_state[i, 2] =  min_value3 

          
        self.publish_pool_state(pool_state)  
        pool_state = np.reshape(pool_state,(540))
        goal_pos = self.transform_goal(self.world_frame_goal, pos, psi)   # roughly (-1, 1) range
        a = goal_pos[0]
        b =  goal_pos[1]
        goal_pos[0] = b
        goal_pos[1] = a
         # ------------------------ change by mingao： waypoint ------------------------  #
        global xxxxxxx
        global yyyyyyy 
        if self.target_point_x  is not None :
            xxxxxxx = self.target_point_x
            yyyyyyy =  self.target_point_y
        goal_pos[0] =  yyyyyyy 
        goal_pos[1] =   xxxxxxx 
        print("goal_pos1",goal_pos )
        if a < 3.50:
            goal_pos[0] = b
            goal_pos[1] = a
        print("correct goal_pos",goal_pos )
        print("current pso",pos )
         # ------------------------ change by mingao： waypoiint ------------------------  #
             
        self.target_point_x = None
        rela_angle = np.arctan2(goal_pos[0],goal_pos[1])                
        rela_dis = np.sqrt(goal_pos[0]**2+goal_pos[1]**2)  
        print("correct rela_angle",rela_angle )
        print("correct rela_dis",rela_dis )
        target_pose = [rela_dis, rela_angle]
        state = np.concatenate([pool_state,target_pose,action_now,[max_action_0,max_action_1, max_acc_0, max_acc_1]], axis=0)
        obs =state
        return obs   
        
        
    def publish_pool_state(self, pool_state):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "pool_state"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        # print(pool_state.shape[0])
        for i in range(pool_state.shape[0]):
            # print(pool_state)
            
            y =   - pool_state[i, 0] * pool_state[i, 2]
            x =   pool_state[i, 1] * pool_state[i, 2]
            z = 0
            p = Point()
            p.x = x
            p.y = y
            p.z = z
            marker.points.append(p)

        self.marker_pub.publish(marker)   
             
    def transform_goal(self, goal_pos, pos, psi):
        """ transform goal in the robot frame
        params:
            pos_1
        """
        R_r2i = np.matrix([[np.cos(psi), -np.sin(psi), pos.x], [np.sin(psi), np.cos(psi), pos.y], [0, 0, 1]])
        R_i2r = np.linalg.inv(R_r2i)
        pi = np.matrix([[goal_pos[0]], [goal_pos[1]], [1]])
        pr = np.matmul(R_i2r, pi)
        lg = np.array([pr[0,0], pr[1, 0]])
        return lg
    
    # def transform_goal(self, goal_pos, pos, psi):
    #         abs_x = goal_pos[0] - pos.x
    #         abs_y = goal_pos[1] - pos.y
            
    #         # ... (计算相对距离和角度)
    #         trans_matrix = np.matrix([[np.cos(psi), np.sin(psi)], [-np.sin(psi), np.cos(psi)]])
    #         rela = np.matmul(trans_matrix,np.array([[abs_x],[abs_y]]))
    #         rela_x = rela[0,0]
    #         rela_y = rela[1,0]
    #         lg = np.array([rela_x , rela_y])
    #         return lg