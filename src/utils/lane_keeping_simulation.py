import copy
import io
import os

import carla
import cv2
import numpy as np
import pygame

from src.controller.pure_pursuit import PurePursuitPlusPID
from src.lane_detector.lane_detector_custom import LaneDetectionHandler
from src.utils.camera_geometry import CameraGeometry
from src.utils.carla_utils_lane import carla_vec_to_np_array, carla_img_to_array, CarlaSyncMode, \
    draw_image_np, should_quit, dist_point_linestring, VehicleStates
from src.utils.vehicle_command import VehicleCommand


class LaneKeepingHandler:
    def __init__(self, fps: int = 10, vehicle_velocity: float = 10, half_image: bool = True,
                 use_lane_detector: bool = True):

        self.controller = PurePursuitPlusPID()  # TODO:to be debuged later 
        self.use_lane_detector = use_lane_detector
        self.lane_detector_handler = None
        self.blueprint_library = None
        self.clock, self.font, self.display = None, None, None
        self.world, self.client = None, None
        self.visualize()

        self.actor_list = []
        self.sensor_list = []
        self.vehicle_list = []
        self.frame = 0
        self.max_error = 0
        self.fps = fps
        self.desired_lane_keeping_speed = vehicle_velocity
        self.init_timestamp = None
        self.half_image = half_image

    def visualize(self):
        print("Initializing real time visualization")
        main_image_shape = (800, 600)
        pygame.init()
        self.display = pygame.display.set_mode(
            main_image_shape,
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.font = pygame.font.SysFont("monospace", 15)
        self.clock = pygame.time.Clock()

    def connect_to_client(self):
        print("Connecting to Unreal Engine Client")
        self.client = carla.Client("127.0.0.1", 2000)
        self.client.set_timeout(70)
        self.world = self.client.get_world()
        if os.path.basename(self.world.get_map().name) != 'Town04':
            self.world: carla.World = self.client.load_world('Town04')

        print("Client Connection Established")

    def weather_parameters(self, predefined_weather=carla.WeatherParameters.ClearNoon):
        self.world.set_weather(predefined_weather)
        print("Weather changed successfully to the desired state.")

    def initialize_simulation(self):
        print("Initializing simulation...")
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()

        ego_vehicle_blueprint = self.blueprint_library.filter('model3')[0]
        self.vehicle = self.world.spawn_actor(
            ego_vehicle_blueprint,
            self.map.get_spawn_points()[90])  # FixMe: Handle spawn point better
        self.actor_list.append(self.vehicle)
        self.vehicle_list.append(self.vehicle)

        # visualization cam (no functionality)
        visualization_rgb_cam = self.world.spawn_actor(
            self.blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-10)),
            attach_to=self.vehicle)
        self.actor_list.append(visualization_rgb_cam)
        self.sensor_list.append(visualization_rgb_cam)

        if self.half_image:
            cg = CameraGeometry(image_width=512, image_height=256)
        else:
            cg = CameraGeometry()

        if self.use_lane_detector is True:
            self.lane_detector_handler = LaneDetectionHandler(cam_geom=cg, model_path="C:\\Users\\behna\\OneDrive\\Dokumente\\My Doccuments\\Pycharm_projects\\CARLA\\src\\lane_detector\\fastai_model.pth")
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

    def get_sensor_data(self, sync_mode, timeout: float = 3):  # Fixme:Tobe decreased later because it slows down simulation
        tick_response = sync_mode.tick(timeout=timeout)
        snapshot, image_rgb, image_windshield = tick_response
        return snapshot, image_rgb, image_windshield

    def exec(self):
        with CarlaSyncMode(self.world, *self.sensor_list, fps=self.fps) as sync_mode:
            while True:
                if should_quit():
                    return
                self.clock.tick()

                # Advance the simulation and wait for the data.
                try:  # Sometimes get_sensor_data does not work at the beginning of the simulation
                    snapshot, image_rgb, image_windshield = self.get_sensor_data(sync_mode)
                    # image_rgb from the camera outside of Auto
                except Exception as error:
                    print(error)
                    continue
                vehicle_command = VehicleCommand()
                speed = np.linalg.norm(carla_vec_to_np_array(self.vehicle.get_velocity()))

                if self.use_lane_detector is True:
                    # if self.frame % 2 == 0:  # FixMe: Why?
                    traj, viz = self.get_trajectory_from_lane_detector(self.lane_detector_handler,
                                                                       image_windshield)
                else:
                    traj = self.get_trajectory_from_map(self.map, self.vehicle)
                traj[:, 1] = -traj[:, 1]
                # vehicle_states = VehicleStates(self.vehicle)
                vehicle_command.throttle, vehicle_command.steer = self.controller.get_control(self.vehicle,
                                                                                              traj,
                                                                                              desired_speed=self.desired_lane_keeping_speed,
                                                                                              dt=1. / self.fps)

                vehicle_command.send_control(self.vehicle)

                self.fps = round(1.0 / snapshot.timestamp.delta_seconds)

                try:
                    dist = dist_point_linestring(np.array([0, 0]), traj)
                    cross_track_error = int(dist * 100)
                    self.max_error = max(self.max_error, cross_track_error)
                except UnboundLocalError:
                    cross_track_error = np.inf
                    self.max_error = np.inf

                # Draw the display.
                image_rgb = copy.copy(carla_img_to_array(image_rgb))
                if self.use_lane_detector is True:
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
                         "max lat. error (cm): {}".format(self.max_error)
                         ]

                for it, t in enumerate(texts):
                    self.display.blit(
                        self.font.render(t, True, (0, 0, 0)), (image_rgb.shape[1] - 270, 20 + dy * it))

                pygame.display.flip()

                self.frame += 1

    def terminate(self):
        print('destroying actors.')
        for actor in self.actor_list:
            actor.destroy()

        pygame.quit()
        print('done.')

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


if __name__ == '__main__':
    lane_keeping_handler = LaneKeepingHandler()
    lane_keeping_handler.connect_to_client()
    lane_keeping_handler.weather_parameters()
    lane_keeping_handler.initialize_simulation()
    try:
        lane_keeping_handler.exec()
    except Exception:
        lane_keeping_handler.terminate()
