import math
import queue
from typing import Tuple, Any, Dict, Union, List

import carla
import numpy as np
import pygame
from spatialmath import SE2

from utils.config_loader import config


def draw_waypoints(world_, waypoints_, road_id=None, life_time=50.0):
    for waypoint in waypoints_:

        if waypoint.road_id == road_id:
            world_.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False,
                                     color=carla.Color(r=0, g=255, b=0), life_time=life_time,
                                     persistent_lines=True)


def filter_waypoints(waypoints_, road_id):
    filtered_waypoints_ = []
    for waypoint in waypoints_:
        if waypoint.road_id == road_id:
            filtered_waypoints_.append(waypoint)
    return filtered_waypoints_


class TrajectoryToFollow:
    def __init__(self, trajectory_index: int = config.CARLA_simulator.FOLLOW_TRAJECTORY.TRAJECTORY_INDEX) -> None:
        self.trajectory_index = trajectory_index

    def get_trajectory_data(self) -> Tuple[Any, list, list]:
        if self.trajectory_index == 0:
            carla_map = config.carla_trajectories.TRAJECTORY_TO_FOLLOW.TRAJECTORY_INDEX_0.MAP
            road_id_list: list = config.carla_trajectories.TRAJECTORY_TO_FOLLOW.TRAJECTORY_INDEX_0.ROAD_ID
            filtered_point_index_list: list = \
                config.carla_trajectories.TRAJECTORY_TO_FOLLOW.TRAJECTORY_INDEX_0.FILTERED_POINT_INDEX
        elif self.trajectory_index == 1:
            carla_map = config.carla_trajectories.TRAJECTORY_TO_FOLLOW.TRAJECTORY_INDEX_1.MAP
            road_id_list: list = config.carla_trajectories.TRAJECTORY_TO_FOLLOW.TRAJECTORY_INDEX_1.ROAD_ID
            filtered_point_index_list: list = \
                config.carla_trajectories.TRAJECTORY_TO_FOLLOW.TRAJECTORY_INDEX_1.FILTERED_POINT_INDEX

        elif self.trajectory_index == 2:
            carla_map = config.carla_trajectories.TRAJECTORY_TO_FOLLOW.TRAJECTORY_INDEX_2.MAP
            road_id_list: list = config.carla_trajectories.TRAJECTORY_TO_FOLLOW.TRAJECTORY_INDEX_2.ROAD_ID
            filtered_point_index_list: list = \
                config.carla_trajectories.TRAJECTORY_TO_FOLLOW.TRAJECTORY_INDEX_2.FILTERED_POINT_INDEX

        elif self.trajectory_index == 3:
            carla_map = config.carla_trajectories.TRAJECTORY_TO_FOLLOW.TRAJECTORY_INDEX_3.MAP
            road_id_list: list = config.carla_trajectories.TRAJECTORY_TO_FOLLOW.TRAJECTORY_INDEX_3.ROAD_ID
            filtered_point_index_list: list = \
                config.carla_trajectories.TRAJECTORY_TO_FOLLOW.TRAJECTORY_INDEX_3.FILTERED_POINT_INDEX

        elif self.trajectory_index == 4:
            carla_map = config.carla_trajectories.TRAJECTORY_TO_FOLLOW.TRAJECTORY_INDEX_4.MAP
            road_id_list: list = config.carla_trajectories.TRAJECTORY_TO_FOLLOW.TRAJECTORY_INDEX_4.ROAD_ID
            filtered_point_index_list: list = \
                config.carla_trajectories.TRAJECTORY_TO_FOLLOW.TRAJECTORY_INDEX_4.FILTERED_POINT_INDEX
        else:
            raise NotImplementedError('A trajectory with index {} has not been implemented.'
                                      .format(self.trajectory_index))
        return carla_map, road_id_list, filtered_point_index_list

    def get_ego_vehicle_spwan_point(self) -> Union[Dict[str, int], Dict[str, int]]:
        if self.trajectory_index == 0:
            ego_spawn_point: Union = \
                {'road_id': config.carla_trajectories.TRAJECTORY_TO_FOLLOW.TRAJECTORY_INDEX_0.
                EGO_VEHICLE_SPAWN_POINT.ROAD_ID,
                 'filtered_points_index': config.carla_trajectories.TRAJECTORY_TO_FOLLOW.TRAJECTORY_INDEX_0.
                 EGO_VEHICLE_SPAWN_POINT.FILTERED_POINT_INDEX}
        elif self.trajectory_index == 1:
            ego_spawn_point: Union = \
                {'road_id': config.carla_trajectories.TRAJECTORY_TO_FOLLOW.TRAJECTORY_INDEX_1.
                EGO_VEHICLE_SPAWN_POINT.ROAD_ID,
                 'filtered_points_index': config.carla_trajectories.TRAJECTORY_TO_FOLLOW.TRAJECTORY_INDEX_1.
                 EGO_VEHICLE_SPAWN_POINT.FILTERED_POINT_INDEX}

        elif self.trajectory_index == 2:
            ego_spawn_point: Union = \
                {'road_id': config.carla_trajectories.TRAJECTORY_TO_FOLLOW.TRAJECTORY_INDEX_2.
                EGO_VEHICLE_SPAWN_POINT.ROAD_ID,
                 'filtered_points_index': config.carla_trajectories.TRAJECTORY_TO_FOLLOW.TRAJECTORY_INDEX_2.
                 EGO_VEHICLE_SPAWN_POINT.FILTERED_POINT_INDEX}

        elif self.trajectory_index == 3:
            ego_spawn_point: Union = \
                {'road_id': config.carla_trajectories.TRAJECTORY_TO_FOLLOW.TRAJECTORY_INDEX_3.
                EGO_VEHICLE_SPAWN_POINT.ROAD_ID,
                 'filtered_points_index': config.carla_trajectories.TRAJECTORY_TO_FOLLOW.TRAJECTORY_INDEX_3.
                 EGO_VEHICLE_SPAWN_POINT.FILTERED_POINT_INDEX}

        elif self.trajectory_index == 4:
            ego_spawn_point: Union = \
                {'road_id': config.carla_trajectories.TRAJECTORY_TO_FOLLOW.TRAJECTORY_INDEX_4.
                EGO_VEHICLE_SPAWN_POINT.ROAD_ID,
                 'filtered_points_index': config.carla_trajectories.TRAJECTORY_TO_FOLLOW.TRAJECTORY_INDEX_4.
                 EGO_VEHICLE_SPAWN_POINT.FILTERED_POINT_INDEX}
        else:
            raise NotImplementedError('A trajectory with index {} has not been implemented.')
        return ego_spawn_point


def carla_vec_to_np_array(vec):
    return np.array([vec.x,
                     vec.y,
                     vec.z])


class CarlaSyncMode(object):
    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


def carla_img_to_array(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array


def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def draw_image_np(surface, image, blend=False):
    array = image
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def dist_point_linestring(p, line_string):
    """ Compute distance between a point and a line_string (a.k.a. polyline)
    """
    a = line_string[:-1, :]
    b = line_string[1:, :]
    return np.min(linesegment_distances(p, a, b))


def linesegment_distances(p, a, b):
    # normalized tangent vectors
    d_ba = b - a
    d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1]).reshape(-1, 1)))

    # signed parallel distance components
    # rowwise dot products of 2D vectors
    s = np.multiply(a - p, d).sum(axis=1)
    t = np.multiply(p - b, d).sum(axis=1)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(s))])

    # perpendicular distance component
    # rowwise cross products of 2D vectors
    d_pa = p - a
    c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]

    return np.hypot(h, c)


def convert_from_global_to_local_reference_frame(global_x: float,
                                                 global_y: float,
                                                 yaw_angle_degree: float) -> Tuple[float, float]:
    global_param = SE2(global_x, global_y, 0)
    rotational_vector = SE2(0, 0, np.radians(yaw_angle_degree))
    local_param = rotational_vector.inv() * global_param
    local_param_x = local_param.xyt()[0]
    local_param_y = local_param.xyt()[1]
    return local_param_x, local_param_y


class VehicleStates:
    def __init__(self, vehicle):
        self.vehicle_transform = vehicle.get_transform()
        self.vehicle_location = self.vehicle_transform.location
        self.vehicle_rotation = self.vehicle_transform.rotation
        # Vehicle Pose:
        self.location_x = self.vehicle_location.x
        self.location_y = self.vehicle_location.y
        self.location_z = self.vehicle_location.z
        self.rotation_roll_deg = self.vehicle_rotation.roll
        self.rotation_pitch_deg = self.vehicle_rotation.pitch
        self.rotation_yaw_deg = self.vehicle_rotation.yaw

        # Vehicle Linear Speed:
        self.linear_velocity = vehicle.get_velocity()  # m/s  # in global frame
        self.linear_velocity_x = self.linear_velocity.x
        self.linear_velocity_y = self.linear_velocity.y
        self.linear_velocity_z = self.linear_velocity.z
        self.overall_linear_velocity = np.linalg.norm(carla_vec_to_np_array(self.linear_velocity))

        # in vehicle reference frame
        self.linear_velocity_x_relative, self.linear_velocity_y_relative = \
            convert_from_global_to_local_reference_frame(self.linear_velocity_x, self.linear_velocity_y,
                                                         self.rotation_yaw_deg)
        self.side_slip_angle: float = math.atan(self.linear_velocity_y_relative / self.linear_velocity_x_relative) \
            if self.linear_velocity_x_relative != 0 else 0

        # create vector from relative speeds
        self.linear_velocity_relative = carla.Vector3D(x=self.linear_velocity_x_relative,
                                                       y=self.linear_velocity_y_relative,
                                                       z=self.linear_velocity_z)

        # Vehicle Angular Speed
        self.angular_velocity = vehicle.get_angular_velocity()  # deg/s
        self.angular_velocity_x = self.angular_velocity.x
        self.angular_velocity_y = self.angular_velocity.y
        self.angular_velocity_z = self.angular_velocity.z
        self.overall_angular_velocity = carla_vec_to_np_array(self.angular_velocity)

        # Vehicle Acceleration
        self.linear_acceleration = vehicle.get_acceleration()  # m/s^2
        self.linear_acceleration_x = self.linear_acceleration.x
        self.linear_acceleration_y = self.linear_acceleration.y
        self.linear_acceleration_z = self.linear_acceleration.z
        self.overall_linear_acceleration = carla_vec_to_np_array(self.linear_acceleration)

        # Vehicle Acceleration in Vehicle Reference Frame
        self.linear_acceleration_x_relative, self.linear_acceleration_y_relative = \
            convert_from_global_to_local_reference_frame(self.linear_acceleration_x, self.linear_acceleration_y,
                                                         self.rotation_yaw_deg)

        self.physics_control = vehicle.get_physics_control()
        self.mass = self.physics_control.mass
        self.center_of_mass = {"cg_x": self.physics_control.center_of_mass.x,
                               "cg_y": self.physics_control.center_of_mass.y,
                               "cg_z": self.physics_control.center_of_mass.z}
        self.distance_cg_to_front_axle: float = config.vehicle.a
        self.distance_cg_to_rear_axle: float = config.vehicle.b
        self.wheelbase: float = self.distance_cg_to_front_axle + self.distance_cg_to_rear_axle
        self.inertia_zz: float = 0.95 * self.mass * (self.wheelbase / 2) ** 2
        self.cornering_stiffness_tyre_FL: float = config.vehicle.cornering_stiffness.tyre_FL
        self.cornering_stiffness_tyre_FR: float = config.vehicle.cornering_stiffness.tyre_FR
        self.cornering_stiffness_tyre_RL: float = config.vehicle.cornering_stiffness.tyre_RL
        self.cornering_stiffness_tyre_RR: float = config.vehicle.cornering_stiffness.tyre_RR
        self.cornering_stiffness_bicycle_model: Union[Dict[str, float], Dict[str, float]] = \
            {"front": self.cornering_stiffness_tyre_FL + self.cornering_stiffness_tyre_FR,
             "rear": self.cornering_stiffness_tyre_RL + self.cornering_stiffness_tyre_RR}


class VehicleStatesROS:
    def __init__(self, location_x, location_y, location_z, rotation_roll_deg, rotation_pitch_deg, rotation_yaw_deg,
                 linear_velocity_x, linear_velocity_y, linear_velocity_z, angular_velocity_x, angular_velocity_y,
                 angular_velocity_z, linear_acceleration_x, linear_acceleration_y, linear_acceleration_z,
                 mass, center_of_mass_x, center_of_mass_y, center_of_mass_z):
        self.location_x = location_x
        self.location_y = location_y
        self.location_z = location_z

        self.rotation_roll_deg = rotation_roll_deg
        self.rotation_pitch_deg = rotation_pitch_deg
        self.rotation_yaw_deg = rotation_yaw_deg

        self.linear_velocity_x = linear_velocity_x
        self.linear_velocity_y = linear_velocity_y
        self.linear_velocity_z = linear_velocity_z
        self.overall_linear_velocity = np.linalg.norm(np.array([linear_velocity_x,
                                                                linear_velocity_y,
                                                                linear_velocity_z]))
        # in vehicle reference frame
        self.linear_velocity_x_relative, self.linear_velocity_y_relative = \
            convert_from_global_to_local_reference_frame(self.linear_velocity_x, self.linear_velocity_y,
                                                         self.rotation_yaw_deg)
        self.side_slip_angle: float = math.atan(self.linear_velocity_y_relative / self.linear_velocity_x_relative) \
            if self.linear_velocity_x_relative != 0 else 0

        self.angular_velocity_x = angular_velocity_x
        self.angular_velocity_y = angular_velocity_y
        self.angular_velocity_z = angular_velocity_z
        self.overall_angular_velocity = np.linalg.norm(np.array([angular_velocity_x,
                                                                 angular_velocity_y,
                                                                 angular_velocity_z]))
        self.linear_acceleration_x = linear_acceleration_x
        self.linear_acceleration_y = linear_acceleration_y
        self.linear_acceleration_z = linear_acceleration_z
        self.overall_linear_acceleration = np.linalg.norm(np.array([linear_acceleration_x,
                                                                    linear_acceleration_y,
                                                                    linear_acceleration_z]))
        self.mass = mass
        self.center_of_mass_x = center_of_mass_x
        self.center_of_mass_y = center_of_mass_y
        self.center_of_mass_z = center_of_mass_z
        self.center_of_mass = {"cg_x": self.center_of_mass_x,
                               "cg_y": self.center_of_mass_y,
                               "cg_z": self.center_of_mass_z}

        self.distance_cg_to_front_axle: float = config.vehicle.a
        self.distance_cg_to_rear_axle: float = config.vehicle.b
        self.wheelbase: float = self.distance_cg_to_front_axle + self.distance_cg_to_rear_axle
        self.inertia_zz: float = 0.95 * self.mass * (self.wheelbase / 2) ** 2
        self.cornering_stiffness_tyre_FL: float = config.vehicle.cornering_stiffness.tyre_FL
        self.cornering_stiffness_tyre_FR: float = config.vehicle.cornering_stiffness.tyre_FR
        self.cornering_stiffness_tyre_RL: float = config.vehicle.cornering_stiffness.tyre_RL
        self.cornering_stiffness_tyre_RR: float = config.vehicle.cornering_stiffness.tyre_RR
        self.cornering_stiffness_bicycle_model: Union[Dict[str, float], Dict[str, float]] = \
            {"front": self.cornering_stiffness_tyre_FL + self.cornering_stiffness_tyre_FR,
             "rear": self.cornering_stiffness_tyre_RL + self.cornering_stiffness_tyre_RR}


class RoadFeatures:
    def __init__(self, waypoints: np.ndarray,
                 target_waypoint_multiplier: float = config.carla_trajectories.RATIO_TO_SELECT_TARGET_WAYPOINT) -> None:
        self.waypoints = waypoints
        self.waypoints_longitudinal = waypoints[:, 0]
        self.waypoints_lateral = waypoints[:, 1]
        self.fitted_polyline = np.polyfit(self.waypoints_longitudinal, self.waypoints_lateral, deg=3)
        self.fitted_line = np.poly1d(self.fitted_polyline)
        self.derivitive_order_1 = np.polyder(self.fitted_line, m=1)
        self.derivitive_order_2 = np.polyder(self.fitted_line, m=2)
        self.derivitive_order_3 = np.polyder(self.fitted_line, m=3)
        self.target_waypoint: List[float] = [waypoints[int(target_waypoint_multiplier * len(waypoints))][0],
                                             waypoints[int(target_waypoint_multiplier * len(waypoints))][1]]

        self.road_curvature: Union[float, Any] = lambda: (1 + self.derivitive_order_1(self.target_waypoint[0]) ** 2) \
                                                         ** 1.5 / (self.derivitive_order_2(self.target_waypoint[0]))
        self.mean_road_curvature_absolute = np.mean(
            [abs((1 + self.derivitive_order_1(waypoint_longitudinal) ** 2) ** 1.5 /
                 (self.derivitive_order_2(waypoint_longitudinal)))
             for waypoint_longitudinal in self.waypoints_longitudinal])

        self.road_curvature_inverse = lambda: 1 / self.road_curvature()
        self.theta_e: Union[float, Any] = lambda: math.atan(self.derivitive_order_1(self.target_waypoint[0]))
        self.e_cg: Union[float, Any] = lambda: self.target_waypoint[1]


def calculate_state_variables(road_params: RoadFeatures, vehicle_states: VehicleStates) -> np.array:
    x0_ = np.array([vehicle_states.side_slip_angle, np.radians(vehicle_states.angular_velocity_z),
                    road_params.theta_e(), road_params.e_cg()]).reshape(-1, 1)
    # if -0.5 < e_cg < -0.5 and -np.radians(20) < theta_e < np.radians(20):  # To prevent oscillation behavior
    #     x0_ = np.array([0, 0, 0, 0]).reshape(-1, 1)
    return x0_
