import numpy as np
import scipy.io
import json

class Camera:
    def __init__(self, path, filetype="MAT"):
        if filetype=="MAT":
            self.readFromMAT(path)
        
        if filetype=="JSON":
            self.readFromJSON(path)

    def readFromMAT(self, path):
        f = scipy.io.loadmat(path)

        self.setIntrinsic(f['cam'][0]['A'][0])
        self.setDistortionCoefficients(f['cam'][0][0]['kc'].squeeze())

    #TODO: Implement
    def readFromJSON(self, path):
        with open(path) as file:
            # Load JSON File
            DICT = json.load(file)

            self.setIntrinsic(np.array(DICT['Intrinsic']))
            self.setDistortionCoefficients(np.array(DICT['DistortionCoefficients']))

    def setIntrinsic(self, mat):
        self._camera_matrix = mat

    def setDistortionCoefficients(self, mat):
        self._distortion_coefficients = mat

    def intrinsic(self):
        return self._camera_matrix

    def distortionCoefficients(self):
        return self._distortion_coefficients

    def getRay(self, point2d):
        homogenous = np.concatenate([point2d, [1.0]])
        homogenous = np.matmul(np.linalg.inv(self._camera_matrix), homogenous)
        return homogenous / np.linalg.norm(homogenous)

    def getRayMat(self, points2d):
        homogenous = np.concatenate([points2d, np.ones((points2d.shape[0], 1))], axis=1)
        homogenous = np.matmul(np.linalg.inv(self._camera_matrix), homogenous.T).T
        return homogenous / np.expand_dims(np.linalg.norm(homogenous, axis=1), -1)

    def projectToNDC(self, points3D):
        points2d = np.matmul(self._camera_matrix, points3D.T).T
        return points2d

    def project(self, points3D):
        points2d = np.matmul(self._camera_matrix, points3D.T).T
        points2d /= np.expand_dims(points2d[:, 2], -1)
        return points2d[:, :2]
