from utils import *


class PerspectiveTransformHelper:

    def __init__(self, difference_percentage_w=0.6, difference_percentage_h=0.6, degrees_range=(5, 30)):
        self.difference_percentage_w = difference_percentage_w
        self.difference_percentage_h = difference_percentage_h
        self.degrees_range = degrees_range
        self.minX = None
        self.maxX = None
        self.minY = None
        self.maxY = None

    @staticmethod
    def get_random(distance, difference_percentage):
        distance_difference = np.random.randint(np.round(distance * 0.2), np.round(distance * difference_percentage))
        return distance_difference // 2, distance - (distance_difference // 2)

    def generate_random_tetrahedron(self, img):
        height, width = img.shape[:2]
        Ax, Bx = self.get_random(width, self.difference_percentage_w)
        Cx, Dx = self.get_random(width, self.difference_percentage_w)
        Ay, Cy = self.get_random(height, self.difference_percentage_h)
        By, Dy = self.get_random(height, self.difference_percentage_h)
        tetrahedron = np.float32([[Ax, Ay], [Bx, By], [Cx, Cy], [Dx, Dy]])
        return tetrahedron

    def rotate_image(self, mat):
        """
        Rotates an image (angle in degrees) and expands image to avoid cropping
        """
        angle = np.random.randint(*self.degrees_range) * np.random.choice([-1, 1])

        height, width = mat.shape[:2]
        image_center = (width / 2,
                        height / 2)

        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)
        rotation_mat[0, 2] += bound_w / 2 - image_center[0]
        rotation_mat[1, 2] += bound_h / 2 - image_center[1]
        rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
        return rotated_mat, rotation_mat, bound_w, bound_h

    def _rotate_polygon(self, pts, M):
        """Transforms a list of points, `pts`,
        using the affine transform `A`."""
        return rotate_polygon(pts, M) - np.array([int(self.minX), int(self.minY)])

    def change_perspective(self, img, rotation=False):
        height, width = img.shape[:2]
        pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        pts2 = self.generate_random_tetrahedron(img)
        if rotation:
            pts2 = np.hstack((pts2, np.ones((4, 1))))
            rotated_mat, rotation_mat, width, height = self.rotate_image(img)
            pts2 = np.float32(np.dot(pts2, rotation_mat.T))
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, M, (width, height))
        self.minX, self.maxX = pts2[:, 0][pts2[:, 0].argsort()][[0, -1]]
        self.minY, self.maxY = pts2[:, 1][pts2[:, 1].argsort()][[0, -1]]
        return dst[int(self.minY):int(self.maxY), int(self.minX):int(self.maxX)], M