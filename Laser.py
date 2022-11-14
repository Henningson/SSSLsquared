import numpy as np
import scipy.io
import math
import json

class Laser:
    def __init__(self, path="", filetype=None):
        if filetype=="MAT":
            self.readFromMAT(path)
        
        if filetype=="JSON":
            self.readFromJSON(path)

    def readFromMAT(self, path):
        f = scipy.io.loadmat(path)
        
        try:
            self.setLaserDimensions(f['lsr']['lsrArrayDims'][0][0][0][0][0][0][0], f['lsr']['lsrArrayDims'][0][0][0][0][1][0][0])
        except:
            self.setLaserDimensions(18, 18)
            print("Couldn't load Laser Dimensions, setting Laser to Size 18 x 18.")
        
        self.setRotationMatrix(f['lsr']['R'][0][0])
        self.setTranslation(f['lsr']['t'][0][0][:, 0])
        self.setAlpha(f['lsr']['alpha'][0][0][0][0])
        self.setLambdas(f['lsr']['Lambda'][0][0])
        self._direction = np.matmul(-self._rotation_matrix, np.array([[0.0, 0.0, -1.0]]).T).T

        self.generateLaserRays()

    #TODO: Implement
    def readFromJSON(self, path):
        with open(path) as file:
            # Load JSON File
            DICT = json.load(file)
            self.setLaserDimensions(DICT['Dimensions'][0], DICT['Dimensions'][1])
            self.setRotationMatrix(np.array(DICT['Rotation']))
            self.setTranslation(np.array(DICT['Translation']))
            self.setAlpha(np.array(DICT['Alpha']))
        
        self._direction = np.matmul(-self._rotation_matrix, np.array([[0.0, 0.0, -1.0]]).T).T
        self.generateLaserRays()

    def generateLaserRays(self):
        laserField = list()

        for x in range(self._gridWidth):
            for y in range(self._gridHeight):
                laserField.append(np.array([np.tan((x - (self._gridWidth/2.0)) * self._alpha), np.tan((y - (self._gridHeight/2.0)) * self._alpha), -1.0]))

        self._laserRays = np.matmul(-self._rotation_matrix, np.stack(laserField).T).T

    def setRays(self, laserRays):
        self._laserRays = laserRays

    def setLaserDimensions(self, gridWidth, gridHeight):
        self._gridWidth = gridWidth
        self._gridHeight = gridHeight

    def setAlpha(self, alpha):
        self._alpha = alpha

    def setRotationMatrix(self, rotMat):
        self._rotation_matrix = rotMat

    def setTranslation(self, translation):
        self._translation = translation

    def direction(self):
        return self._direction

    #TODO: Implement
    def setLambdas(self, lambdas):
        self._lambdas = lambdas

    def translation(self):
        return self._translation

    def origin(self):
        return self._translation

    def rotationMatrix(self):
        return self._rotation_matrix
    
    def alpha(self):
        return self._alpha

    def gridHeight(self):
        return self._gridHeight

    def gridWidth(self):
        return self._gridWidth

    def getDims(self):
        return np.array([self._gridWidth, self._gridHeight])

    def rays(self):
        return self._laserRays

    def lambdas(self):
        return self._lambdas

    def getXYfromN(self, n):
        return math.floor(n/self._gridHeight), n % self._gridWidth

    def getNfromXY(self, x, y):
        return x * self._gridHeight + y

    def ray(self, x, y=None):
        if y:
            return self._laserRays[self.getNfromXY(x, y)]
        
        return self._laserRays[x]