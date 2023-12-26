import cv2
import numpy as np
import torch
import os
import time

from src.utils.camera_geometry import CameraGeometry


class LaneDetectionHandler:
    def __init__(self, cam_geom=CameraGeometry(), model_path: str = "fastai_model.pth",
                 processing_device: str = "cpu") -> None:
        self.cg = cam_geom
        self.cut_v, self.grid = self.cg.precompute_grid()
        self.model_path = model_path
        self.processing_device = processing_device

    def load(self):
        start_time = time.time()
        if self.processing_device == "gpu":
            print("Lane detector handler module is taking advantage of GPU!")
            self.device = "cuda"
            self.model = torch.load(self.model_path).to(self.device)
        elif self.processing_device == "cpu":
            print("Lane detector handler is not utilizing GPU!")
            self.model = torch.load(self.model_path, map_location=torch.device("cpu"))
            self.device = "cpu"
        else:
            if torch.cuda.is_available():
                print("Lane detector handler module is taking advantage of GPU!")
                self.device = "cuda"
                self.model = torch.load(self.model_path).to(self.device)
            else:
                print("Lane detector handler is not utilizing GPU!")
                self.model = torch.load(self.model_path, map_location=torch.device("cpu"))
                self.device = "cpu"
        print("Loading lane detector model took %.2f seconds" % (time.time() - start_time))
        self.model.eval()

    @staticmethod
    def read_image_file_to_array(img_path: str) -> np.ndarray:
        image: np.ndarray = cv2.imread(img_path)
        image: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def detect_from_file(self, img_path: str):
        img_array = self.read_image_file_to_array(img_path)
        background_probability_map, left_lane_probability_map, right_lane_probability_map = self.detect(img_array)
        return background_probability_map, left_lane_probability_map, right_lane_probability_map

    def __predict__(self, img_):
        with torch.no_grad():
            image_tensor = img_.transpose(2, 0, 1).astype('float32') / 255
            x_tensor = torch.from_numpy(image_tensor).to(self.device).unsqueeze(0)
            model_output = torch.softmax(self.model.forward(x_tensor), dim=1).cpu().numpy()
        return model_output

    def detect(self, img_array):
        model_output = self.__predict__(img_array)
        background, left, right = model_output[0, 0, :, :], model_output[0, 1, :, :], model_output[0, 2, :, :]
        return background, left, right

    def fit_poly(self, probs):
        probs_flat = np.ravel(probs[self.cut_v:, :])
        mask = probs_flat > 0.3
        if mask.sum() > 0:
            fitted_polyline_coefficients = np.polyfit(self.grid[:, 0][mask], self.grid[:, 1][mask],
                                                      deg=3, w=probs_flat[mask])
        else:
            fitted_polyline_coefficients = np.array([0., 0., 0., 0.])  # 4 values since poly is of degree 3
        return np.poly1d(fitted_polyline_coefficients)

    def __call__(self, image):
        if isinstance(image, str):
            image = self.read_image_file_to_array(image)
        left_poly, right_poly, _, _ = self.get_fit_and_probs(image)
        return left_poly, right_poly

    def get_fit_and_probs(self, rgb_img_, visualize_segmentation_output: bool = True):
        _, left_lane_probability_map, right_lane_probability_map = self.detect(rgb_img_)
        left_poly = self.fit_poly(left_lane_probability_map)
        right_poly = self.fit_poly(right_lane_probability_map)
        if visualize_segmentation_output:
            bgr_img: np.ndarray = cv2.cvtColor(rgb_img_, cv2.COLOR_RGB2BGR)
            self.get_visualized_segmentation_output(bgr_img, left_lane_probability_map, right_lane_probability_map)
        return left_poly, right_poly, left_lane_probability_map, right_lane_probability_map

    @staticmethod
    def get_visualized_segmentation_output(img_bgr: np.ndarray,
                                           left_lane_probability_map,
                                           right_lane_probability_map) -> None:
        img_height, img_width = img_bgr.shape[:2]
        left_line_mask = left_lane_probability_map > 0.3
        right_line_mask = right_lane_probability_map > 0.3
        prediction_mask = np.zeros((img_height, img_width, 3), np.uint8)
        for width_index in range(img_width):
            for height_index in range(img_height):
                if left_line_mask[height_index][width_index]:
                    prediction_mask[height_index][width_index] = (255, 0, 0)
                elif right_line_mask[height_index][width_index]:
                    prediction_mask[height_index][width_index] = (0, 0, 255)
        visualized_img: np.ndarray = ((0.4 * img_bgr) + (0.6 * prediction_mask)).astype("uint8")
        cv2.imshow("lane_detection_output", visualized_img)
        cv2.waitKey(10)
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    lane_detection_handler = LaneDetectionHandler()
    lane_detection_handler.load()
    IMG_PATH: str = os.path.join(os.path.dirname(__file__), "rgb_cam", "000046.jpg")
    img: np.ndarray = cv2.imread(IMG_PATH)
    rgb_img: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    left_poly_, right_poly_ = lane_detection_handler(rgb_img)
