import ast
import copy
import os
import time

import carla
import cv2
import numpy as np
import pandas as pd
import pygame
import rospy

from consts.consts import NAN

from core.logger import logger, TERMINAL_COLOR_GREEN
from core.vehicle_command import VehicleCommand
from lane_detection.camera_geometry import CameraGeometry
from lane_detection.lane_detector import LaneDetectionHandler
from remote_controllers.remote_controller_handler import remote_controls
from simulation.CARLA.carla_ros_handler import CARLAROSHandler
from simulation.CARLA.carla_utils.carla_utils import carla_vec_to_np_array, carla_img_to_array, CarlaSyncMode, \
    draw_image_np, should_quit, dist_point_linestring, VehicleStates, calculate_state_variables, RoadFeatures
from simulation.CARLA.carla_utils.ros_utils import Actor
from utils.config_loader import config


class LaneKeepingHandler:
    def __init__(self, use_lane_detector: bool = config.lane_keeping.USE_LANE_DETECTION,
                 half_image: bool = config.lane_keeping.USE_HALF_IMAGE,
                 real_time_visualization: bool = config.lane_keeping.VISUALIZE,
                 controller: str = config.control.CONTROLLER):

        self.steer_oscillation_count = 0
        self.prev_steer = None
        self.STEP_MULTIPLIER = 0.05
        self.speed_multiplier = 0.8
        self.autonomous_lane_keeping = False
        self.idle_mode = True
        self.reverse_mode = False

        self.lane_detector_handler = None
        self.blueprint_library = None
        self.clock, self.font, self.display = None, None, None
        self.world, self.client = None, None
        self.use_hil_control = config.general.USE_HARDWARE_IN_THE_LOOP
        self.use_lane_detector = use_lane_detector
        self.half_image = half_image
        self.real_time_visualization = real_time_visualization
        self.actor_data_handler = Actor()
        self.controller = controller
        if self.controller == "pure_pursuit":
            from src.control.pure_pursuit import PurePursuitPlusPID
            self.controller = PurePursuitPlusPID()
        elif self.controller == "mpc":
            from src.control.mpc import MPCPlusPIDModelBased
            self.controller = MPCPlusPIDModelBased()
        # elif self.controller == "pid":  # TODO: pid controller is not implemented yet!
        #     from control.pid import PIDLongitudinalLateralController
        #     self.controller = PIDLongitudinalLateralController()
        else:
            raise ValueError("Unsupported controller: {}".format(self.controller))

        if self.real_time_visualization:
            self.visualize()

        self.actor_list = []
        self.sensor_list = []
        self.vehicle_list = []
        self.frame = 0
        self.max_error = 0
        self.fps = config.CARLA_simulator.FPS
        self.desired_lane_keeping_speed: float = config.CARLA_simulator.FOLLOW_TRAJECTORY.DESIRED_SPEED

        self.ros_handler = None
        if self.use_hil_control:
            from simulation.CARLA.hil.ros_topic_communication import LocalDeviceHILDataTransfer
            self.local_device_hil_data_transfer = LocalDeviceHILDataTransfer()

        if config.general.RECORD_ROSBAG_DATA is True:
            from core.rosbag_recorder import rosbag_start_record
            rosbag_start_record()

        index = 0
        while True:
            self.data_path = os.path.join("data", f"simulation_{index}_controller_{config.control.CONTROLLER}_speed_"
                                                  f"{config.CARLA_simulator.FOLLOW_TRAJECTORY.DESIRED_SPEED}")
            if os.path.exists(self.data_path) and len(os.listdir(self.data_path)) > 0:
                index += 1
            else:
                break
        logger.info(f"Saving simulation results in directory: {self.data_path}")
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        self.dataframe = pd.DataFrame({})
        self.init_timestamp = None

    def visualize(self):
        logger.info("Initializing real time visualization", color=TERMINAL_COLOR_GREEN)
        main_image_shape = (800, 600)
        pygame.init()
        self.display = pygame.display.set_mode(
            main_image_shape,
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.font = pygame.font.SysFont("monospace", 15)
        self.clock = pygame.time.Clock()

    def connect_to_client(self):
        logger.info("Connecting to Unreal Engine Client", color=TERMINAL_COLOR_GREEN)
        self.client = carla.Client(config.CARLA_simulator.CLIENT.IP, config.CARLA_simulator.CLIENT.PORT)
        self.client.set_timeout(config.CARLA_simulator.CLIENT.TIMEOUT)
        self.world = self.client.get_world()
        if os.path.basename(self.world.get_map().name) != config.CARLA_simulator.CLIENT.MAP:
            self.world: carla.World = self.client.load_world(config.CARLA_simulator.CLIENT.MAP)
        logger.info("Clinet Connection Established", color=TERMINAL_COLOR_GREEN)
        self.ros_handler = CARLAROSHandler(self.world)

    def weather_parameters(self, cloudyness=config.CARLA_simulator.WEATHER.CLOUDYNESS,
                           precipitation=config.CARLA_simulator.WEATHER.PRECIPITATION,
                           sun_altitude_angle=config.CARLA_simulator.WEATHER.SUN_ALTITUDE_ANGLE,
                           predefined_weather=eval(config.CARLA_simulator.WEATHER.PREDEFINED_WEATHER)):
        if ast.literal_eval(cloudyness) is None or \
                ast.literal_eval(precipitation) is None or \
                ast.literal_eval(sun_altitude_angle) is None:
            logger.info("Weather parameters are None in the config file. Utilizing predefined weather...",
                        color=TERMINAL_COLOR_GREEN)
            self.world.set_weather(predefined_weather)
        else:
            logger.info("Setting the weather based on the specific values in the config file.",
                        color=TERMINAL_COLOR_GREEN)
            weather = carla.WeatherParameters(cloudyness=cloudyness,
                                              precipitation=precipitation,
                                              sun_altitude_angle=sun_altitude_angle)
            self.world.set_weather(weather)
        logger.info("Weather changed successfully to the desired state.", color=TERMINAL_COLOR_GREEN)

    def initialize_simulation(self):
        logger.info("Initializing simulation...", color=TERMINAL_COLOR_GREEN)
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()

        ego_vehicle_blueprint = self.blueprint_library.filter(config.CARLA_simulator.VEHICLE.MODEL)[0]
        self.vehicle = self.world.spawn_actor(
            ego_vehicle_blueprint,
            self.map.get_spawn_points()[75])  # FixMe: Handle spawn point better
        self.actor_list.append(self.vehicle)
        self.vehicle_list.append(self.vehicle)

        self.vehicle.show_debug_telemetry(True)
        self.world.show_vehicle_telemetry = True

        if self.real_time_visualization:
            # visualization cam (no functionality)
            visualization_rgb_cam = self.world.spawn_actor(
                self.blueprint_library.find('sensor.camera.rgb'),
                carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-10)),
                attach_to=self.vehicle)
            self.actor_list.append(visualization_rgb_cam)
            self.sensor_list.append(visualization_rgb_cam)

        if self.use_lane_detector is True:
            if self.half_image:
                cg = CameraGeometry(image_width=512, image_height=256)
            else:
                cg = CameraGeometry()
            if self.use_hil_control is False:
                self.lane_detector_handler = LaneDetectionHandler(cam_geom=cg)
                self.lane_detector_handler.load()
            # windshield cam
            cam_windshield_transform = carla.Transform(carla.Location(x=0.5, z=cg.height),
                                                       carla.Rotation(pitch=cg.pitch_deg))
            rgb_camera_blueprint = self.blueprint_library.find('sensor.camera.rgb')
            fov = cg.field_of_view_deg
            rgb_camera_blueprint.set_attribute('image_size_x', str(cg.image_width))
            rgb_camera_blueprint.set_attribute('image_size_y', str(cg.image_height))
            rgb_camera_blueprint.set_attribute('fov', str(fov))
            camera_windshield = self.world.spawn_actor(
                rgb_camera_blueprint,
                cam_windshield_transform,
                attach_to=self.vehicle)
            self.actor_list.append(camera_windshield)
            self.sensor_list.append(camera_windshield)

        remote_controls.start()
        if self.use_hil_control is True:
            self.local_device_hil_data_transfer.start()
        logger.info("Simulation initialized successfully.", color=TERMINAL_COLOR_GREEN)

    def get_sensor_data(self, sync_mode, timeout: float = 0.5):
        tick_response = sync_mode.tick(timeout=timeout)
        if self.use_lane_detector:
            snapshot, image_rgb, image_windshield = tick_response
        else:
            snapshot, image_rgb = tick_response
            image_windshield = None

        return snapshot, image_rgb, image_windshield

    def exec(self):
        with CarlaSyncMode(self.world, *self.sensor_list, fps=self.fps) as sync_mode:
            while True:
                if should_quit():
                    return
                if self.real_time_visualization:
                    self.clock.tick()

                # Advance the simulation and wait for the data.
                try:  # Sometimes get_sensor_data does not work at the beginning of the simulation
                    snapshot, image_rgb, image_windshield = self.get_sensor_data(sync_mode)
                except Exception as e:
                    logger.error(e)
                    continue
                vehicle_command = VehicleCommand()

                speed = np.linalg.norm(carla_vec_to_np_array(self.vehicle.get_velocity()))
                tic_computational_time = time.time()
                if self.autonomous_lane_keeping is True:
                    if self.use_hil_control is False:
                        if self.use_lane_detector:
                            if self.frame % 2 == 0:  # FixMe: Why?
                                traj, viz = self.get_trajectory_from_lane_detector(self.lane_detector_handler,
                                                                                   image_windshield)
                        else:
                            traj = self.get_trajectory_from_map(self.map, self.vehicle)
                        traj[:, 1] = -traj[:, 1]
                        vehicle_states = VehicleStates(self.vehicle)
                        vehicle_command.throttle, vehicle_command.steer = self.controller.get_control(self.vehicle,
                                                                                                      traj,
                                                                                                      desired_speed=self.desired_lane_keeping_speed,
                                                                                                      dt=1. / self.fps)

                        # detect if vehicle_command.steer is oscillating
                        if self.prev_steer is not None:
                            if np.abs(vehicle_command.steer - self.prev_steer) > 0.02:
                                self.steer_oscillation_count += 1
                            else:
                                self.steer_oscillation_count = 0
                            if self.steer_oscillation_count > 10:
                                self.steer_oscillation_count = 0
                                dummy = 1
                        self.prev_steer = vehicle_command.steer

                        try:
                            x0 = calculate_state_variables(RoadFeatures(traj), vehicle_states)
                            e_cg, theta_e = x0[3][0], np.rad2deg(x0[2][0])
                            x_target, y_target = RoadFeatures(traj).target_waypoint[0], \
                                RoadFeatures(traj).target_waypoint[1]
                            target_point = [[x_target, y_target]]  # FixMe: is -y_target correct?
                        except Exception:
                            e_cg, theta_e, x_target, y_target, target_point = None, None, None, None, None

                        # FixMe: To remove

                    else:  # Hardware in the loop: VehicleCommand should be received by subscribing to rostopic
                        if self.use_lane_detector is False:
                            traj = self.get_trajectory_from_map(self.map, self.vehicle)
                            traj[:, 1] = -traj[:, 1]
                            image_windshield = None
                        else:
                            traj = None
                        # FixMe: we should give time to other device to process and publish outputs
                        while True:
                            self.local_device_hil_data_transfer.publish(windshield_img=image_windshield,
                                                                        trajectory=traj,
                                                                        vehicle_states=VehicleStates(self.vehicle))
                            # self.local_device_hil_data_transfer.subscribe()
                            if self.local_device_hil_data_transfer.vehicle_command.throttle != 0 or \
                                    self.local_device_hil_data_transfer.vehicle_command.steer != 0:
                                break
                        vehicle_command = self.local_device_hil_data_transfer.get_vehicle_command()
                        logger.info(f"throttle: {vehicle_command.throttle}, steer: {vehicle_command.steer}")
                        self.local_device_hil_data_transfer.reset_vehicle_command()
                        try:
                            e_cg, theta_e = self.local_device_hil_data_transfer.e_cg, self.local_device_hil_data_transfer.theta_e
                            x_target, y_target = self.local_device_hil_data_transfer.x_target, \
                                self.local_device_hil_data_transfer.y_target
                            target_point = [[x_target, y_target]]  # FixMe: is -y_target correct?
                        except Exception:
                            e_cg, theta_e, x_target, y_target, target_point = None, None, None, None, None

                computational_time = time.time() - tic_computational_time
                vehicle_command = self.get_remote_controller_commands(vehicle_command)
                vehicle_command.reverse = self.reverse_mode
                vehicle_command.send_control(self.vehicle)
                vehicle_velocity = {"vehicle_linear_velocity": VehicleStates(self.vehicle).linear_velocity_relative,
                                    "vehicle_angular_velocity": self.vehicle.get_angular_velocity()}
                try:
                    vehicle_waypoint = self.map.get_waypoint(self.vehicle.get_location())
                    right_lane_wp = vehicle_waypoint.get_right_lane()
                    left_lane_wp = vehicle_waypoint.get_left_lane()
                    # FixMe: should we also convert angular velocity to relative?
                    self.ros_handler.vehicle_states(vehicle_command.throttle, vehicle_command.brake,
                                                    vehicle_command.steer,
                                                    e_cg, np.degrees(theta_e),
                                                    np.radians(VehicleStates(self.vehicle).angular_velocity_z),
                                                    x_target, y_target)
                    self.ros_handler.cmd_vel(vehicle_velocity)
                    self.ros_handler.map(self.world.get_map())
                    self.ros_handler.odom(odom_pose_data=self.actor_data_handler.get_current_ros_pose(self.vehicle),
                                          odom_twist_data=self.actor_data_handler.get_current_ros_twist_rotated(
                                              self.vehicle))
                    self.ros_handler.tf(self.actor_data_handler.get_current_ros_transform(self.vehicle))
                    self.ros_handler.marker_array([self.vehicle])

                    timestamp = rospy.Time.now()
                    # convert timestamp to seconds
                    timestamp = timestamp.secs + timestamp.nsecs * 1e-9
                    if self.init_timestamp is None:
                        self.init_timestamp = timestamp
                    simulation_timestamp = timestamp - self.init_timestamp
                    dataframe_msg = {
                        "ROStimestamp": timestamp,
                        "simulation_timestamp": simulation_timestamp,
                        "pose_x": vehicle_states.location_x,
                        "pose_y": vehicle_states.location_y,
                        "pose_z": vehicle_states.location_z,
                        "orien_roll": vehicle_states.rotation_roll_deg,
                        "orien_pitch": vehicle_states.rotation_pitch_deg,
                        "orien_yaw": vehicle_states.rotation_yaw_deg,
                        "computational_time": computational_time,
                        "linear_velocity_x": vehicle_states.linear_velocity_x_relative,
                        "linear_velocity_y": vehicle_states.linear_velocity_y_relative,
                        "angular_velocity_z": vehicle_states.angular_velocity_z,  # deg/s
                        "linear_acceleration_x": vehicle_states.linear_acceleration_x_relative,
                        "linear_acceleration_y": vehicle_states.linear_acceleration_y_relative,
                        "steering_angle_deg": np.rad2deg(vehicle_command.steer * 1.22),  # FixMe: check the 1.22 magic number
                        "y_error": e_cg,
                        "heading_angle_error_deg": theta_e,
                        "side_slip_angle_deg": np.rad2deg(vehicle_states.side_slip_angle),
                        "right_lane_pose_x": right_lane_wp.transform.location.x,
                        "right_lane_pose_y": right_lane_wp.transform.location.y,
                        "left_lane_pose_x": left_lane_wp.transform.location.x,
                        "left_lane_pose_y": left_lane_wp.transform.location.y,
                    }

                    self.dataframe = self.dataframe.append(dataframe_msg, ignore_index=True)
                    self.dataframe.to_csv(os.path.join(self.data_path, "data.csv"), index=False)

                    if target_point is not None:
                        for i in range(len(traj)):
                            target_point.append([traj[i][0], traj[i][1]])
                        self.ros_handler.points(point=target_point)
                except Exception:
                    pass

                self.fps = round(1.0 / snapshot.timestamp.delta_seconds)

                try:
                    dist = dist_point_linestring(np.array([0, 0]), traj)
                    cross_track_error = int(dist * 100)
                    self.max_error = max(self.max_error, cross_track_error)
                except UnboundLocalError:
                    cross_track_error = NAN
                    self.max_error = NAN

                # Draw the display.
                image_rgb = copy.copy(carla_img_to_array(image_rgb))
                if self.use_lane_detector:
                    viz = cv2.resize(viz, (400, 200), interpolation=cv2.INTER_AREA)
                    image_rgb[0:viz.shape[0], 0:viz.shape[1], :] = viz
                    # white background for text
                    image_rgb[10:130, -280:-10, :] = [255, 255, 255]
                draw_image_np(self.display, image_rgb)

                # draw txt
                dy = 20
                texts = ["FPS (real):          {:.2f}".format(float(self.clock.get_fps())),
                         "FPS (simulated):     {}".format(self.fps),
                         "speed (m/s):         {:.2f}".format(speed),
                         "lateral error (cm):  {}".format(cross_track_error),
                         "max lat. error (cm): {}".format(self.max_error),
                         "controller: {}".format(config.control.CONTROLLER),
                         "reverse: {}".format(vehicle_command.reverse),
                         "HIL: {}".format(self.use_hil_control),
                         ]

                for it, t in enumerate(texts):
                    self.display.blit(
                        self.font.render(t, True, (0, 0, 0)), (image_rgb.shape[1] - 270, 20 + dy * it))

                pygame.display.flip()

                self.frame += 1

    def terminate(self):
        logger.info('destroying actors.', color=TERMINAL_COLOR_GREEN)
        for actor in self.actor_list:
            actor.destroy()

        pygame.quit()
        logger.info('done.', color=TERMINAL_COLOR_GREEN)

    def get_trajectory_from_lane_detector(self, ld, image):
        # get lane boundaries using the lane detector
        img = carla_img_to_array(image)
        poly_left, poly_right, left_mask, right_mask = ld.get_fit_and_probs(img)
        # trajectory to follow is the mean of left and right lane boundary
        # note that we multiply with -0.5 instead of 0.5 in the formula for y below
        # according to our lane detector x is forward and y is left, but
        # according to Carla x is forward and y is right.
        x = np.arange(-2, 60, 1.0)
        y = -0.5 * (poly_left(x) + poly_right(x))
        # x,y is now in coordinates centered at camera, but camera is 0.5 in front of vehicle center
        # hence correct x coordinates
        x += 0.5
        traj = np.stack((x, y)).T
        return traj, self.ld_detection_overlay(img, left_mask, right_mask)

    @staticmethod
    def ld_detection_overlay(image, left_mask, right_mask):
        res = copy.copy(image)
        res[left_mask > 0.5, :] = [0, 0, 255]
        res[right_mask > 0.5, :] = [255, 0, 0]
        return res

    @staticmethod
    def get_trajectory_from_map(m, vehicle):
        # get 80 waypoints each 1m apart. If multiple successors choose the one with lower waypoint.id
        wp = m.get_waypoint(vehicle.get_transform().location)
        wps = [wp]
        for _ in range(20):
            next_wps = wp.next(1.0)
            if len(next_wps) > 0:
                wp = sorted(next_wps, key=lambda x: x.id)[0]
            wps.append(wp)

        # transform waypoints to vehicle ref frame
        traj = np.array(
            [np.array([*carla_vec_to_np_array(x.transform.location), 1.]) for x in wps]
        ).T
        trafo_matrix_world_to_vehicle = np.array(vehicle.get_transform().get_inverse_matrix())

        traj = trafo_matrix_world_to_vehicle @ traj
        traj = traj.T
        traj = traj[:, :2]
        return traj

    def get_remote_controller_commands(self, vehicle_command):
        if remote_controls.get_speed_up() is True:
            self.speed_multiplier += self.STEP_MULTIPLIER
            self.speed_multiplier = np.clip(self.speed_multiplier, 0.1, 1.0)
            self.desired_lane_keeping_speed = \
                self.speed_multiplier * config.CARLA_simulator.FOLLOW_TRAJECTORY.DESIRED_SPEED
            logger.info("Joystick speed multiplier : {:3f}".format(self.speed_multiplier))
            logger.info(f"Desired speed for autonomous navigation : {self.desired_lane_keeping_speed}")
        elif remote_controls.get_speed_down() is True:
            self.speed_multiplier -= self.STEP_MULTIPLIER
            self.speed_multiplier = np.clip(self.speed_multiplier, 0.1, 1.0)
            self.desired_lane_keeping_speed = \
                self.speed_multiplier * config.CARLA_simulator.FOLLOW_TRAJECTORY.DESIRED_SPEED
            logger.info("Joystick speed multiplier : {:3f}".format(self.speed_multiplier))
            logger.info(f"Desired speed for autonomous navigation : {self.desired_lane_keeping_speed}")
        if remote_controls.get_throttle() != 0:
            if remote_controls.get_throttle() > 0:
                vehicle_command.throttle = remote_controls.get_throttle() * self.speed_multiplier
            else:
                vehicle_command.brake = abs(remote_controls.get_throttle()) * self.speed_multiplier
        if remote_controls.get_steering() != 0:
            vehicle_command.steer = remote_controls.get_steering() * self.speed_multiplier * 0.4

        if remote_controls.get_reverse() is True:
            self.reverse_mode = not self.reverse_mode

        if remote_controls.get_lane_keeping_mode() is True:
            self.autonomous_lane_keeping = True
            logger.info("Lane keeping mode : {}".format(self.autonomous_lane_keeping))

        if remote_controls.get_idle_mode() is True:
            self.autonomous_lane_keeping = False
            self.use_hil_control = False
            logger.info("Lane keeping mode : {}".format(self.autonomous_lane_keeping))

        if remote_controls.get_hil() is True:
            self.use_hil_control = not self.use_hil_control
            if self.use_hil_control is True and self.autonomous_lane_keeping is False:
                self.autonomous_lane_keeping = True
            logger.info("Hardware in the loop mode : {}".format(self.use_hil_control))

        return vehicle_command


if __name__ == '__main__':
    lane_keeping_handler = LaneKeepingHandler(use_lane_detector=config.lane_keeping.USE_LANE_DETECTION,
                                              half_image=config.lane_keeping.USE_HALF_IMAGE,
                                              real_time_visualization=config.lane_keeping.VISUALIZE,
                                              controller=config.control.CONTROLLER)
    lane_keeping_handler.connect_to_client()
    lane_keeping_handler.weather_parameters()
    lane_keeping_handler.initialize_simulation()
    try:
        lane_keeping_handler.exec()
    except KeyboardInterrupt:
        lane_keeping_handler.terminate()
