import torch

class Network:
    
    def __init__(self, numberOfNodes, ssBreadth, ssLength):
        
        self.ssLength = ssLength
        self.ssBreadth = ssBreadth
        self.numberOfNodes = numberOfNodes
        self.weights = torch.rand(4, numberOfNodes)
        self.nodeCoordinate = torch.rand(numberOfNodes, 2)*min(ssLength, ssBreadth)
        self.ballCoordinate = self.nodeCoordinate[0]
        self.distMatrix = self.computeInterNodeDist()
        
    def computeInterNodeDist(self):
        distMatrix = float("inf")*torch.ones((self.numberOfNodes, self.numberOfNodes))
        
        for i in range(self.numberOfNodes):
            distCoord = self.nodeCoordinate - self.nodeCoordinate[i]
            for j in range(self.numberOfNodes):
                distMatrix[i][j] = torch.sqrt(torch.tensor(distCoord[j][0]**2 + distCoord[j][1]**2, dtype=torch.float32))
                
        return distMatrix

    def computeInput(self):
        self.inputVector = torch.zeros((1, self.numberOfNodes), dtype=torch.float32)
        
        for i in range(self.numberOfNodes):
            try:
                self.inputVector[0][i] = self.distMatrix[i][i+1]
            except IndexError:
                self.inputVector[0][i] = self.distMatrix[i][0]

    def nodeToBallDist(self, ballCoordinate=None):
        if ballCoordinate == None:
            ballCoordinate = self.ballCoordinate
        return torch.sqrt(torch.sum((self.nodeCoordinate - ballCoordinate)**2, axis=1))

    def forwardProp(self):
        z = self.weights@torch.transpose(self.inputVector, 0, 1) 
        a = min(self.ssLength, self.ssBreadth)*torch.sigmoid(z)

        point1 = torch.tensor([a[0], a[1]])
        point2 = torch.tensor([a[2], a[3]])

        distVector1 = self.nodeToBallDist(ballCoordinate=point1)
        distVector2 = self.nodeToBallDist(ballCoordinate=point2)
        distVector = torch.cat((distVector1, distVector2))
        minDist = torch.min(distVector)
        minDistIndex = torch.where(distVector == minDist)[0].tolist()[0]

        if minDistIndex < len(distVector)/2:
            self.ballCoordinate = point1
        else:
            self.ballCoordinate = point2

    '''
    def updateNodeCoord(self):
        self.nodeToBallDist()
    '''

net = Network(4, 3, 3)
net.computeInput()
print(net.ballCoordinate)
print(net.nodeCoordinate)
net.forwardProp()
print(net.ballCoordinate)
print(net.nodeCoordinate)