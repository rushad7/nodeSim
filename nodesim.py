import torch

class Network:
    
    def __init__(self, numberOfNodes, ssBreadth, ssLength):
        
        self.numberOfNodes = numberOfNodes
        self.weights = torch.rand(4, numberOfNodes**2)
        self.nodeCoordinate = torch.rand(numberOfNodes, 2)*min(ssLength, ssBreadth)
        self.distMatrix = self.computeDist(self.nodeCoordinate)
        
    def computeDist(self, cooridnateVector):
        distMatrix = float("inf")*torch.ones((self.numberOfNodes, self.numberOfNodes))
        
        for i in range(self.numberOfNodes):
            distCoord = cooridnateVector - cooridnateVector[i]
            for j in range(self.numberOfNodes):
                distMatrix[i][j] = torch.sqrt(torch.tensor(distCoord[j][0]**2 + distCoord[j][1]**2, dtype=torch.float32))
                
        return distMatrix

    def computeInput(self):
        inputVector = torch.zeros((1, self.numberOfNodes))
        
        for i in range(self.numberOfNodes):
            try:
                inputVector[0][i] = self.distMatrix[i][i+1]
            except IndexError:
                inputVector[0][i] = self.distMatrix[i][0]
        
        return inputVector