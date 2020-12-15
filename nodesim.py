import torch

class Network:
    
    def __init__(self, numberOfNodes, ssBreadth, ssLength):
        
        self.ssLength = ssLength
        self.ssBreadth = ssBreadth
        self.goalCoordinate = torch.tensor([self.ssLength, self.ssBreadth])
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
        self.a = min(self.ssLength, self.ssBreadth)*torch.sigmoid(z)

    def updateBallCoord(self):
        point1 = torch.tensor([self.a[0], self.a[1]])
        point2 = torch.tensor([self.a[2], self.a[3]])

        distVector1 = self.nodeToBallDist(ballCoordinate=point1)
        distVector2 = self.nodeToBallDist(ballCoordinate=point2)
        distVector = torch.cat((distVector1, distVector2))
        minDist = torch.min(distVector1)
        minDistIndex = torch.where(distVector == minDist)[0].tolist()[0]

        if minDistIndex < len(distVector)/2:
            self.ballCoordinate = point1
            self.a = point1
        else:
            self.ballCoordinate = point2
            self.a = point2

    def updateNodeCoord(self):
        distVector = self.nodeToBallDist()
        minDist = torch.min(distVector)
        minDistIndex = torch.where(distVector == minDist)[0].tolist()[0]
        self.nodeCoordinate[minDistIndex] = self.ballCoordinate

    def backProp(self):
        
        def gradJW(grad_JG, grad_GZ, grad_JW):
            grad_JW = grad_JG * torch.transpose(grad_GZ, 0, 1) @ grad_JW
            return  grad_JW
        
        def gradJG(axis):

            if axis == "x":
                grad_JG_num = self.goalCoordinate[0] - self.ballCoordinate[0]
            else:
                grad_JG_num = self.goalCoordinate[1] - self.ballCoordinate[1]
            
            xdiff_sqr = torch.pow(self.goalCoordinate[0] - self.ballCoordinate[0], 2)
            ydiff_sqr = torch.pow(self.goalCoordinate[1] - self.ballCoordinate[1], 2)
            grad_JG_den = torch.sqrt(xdiff_sqr + ydiff_sqr)

            grad_JG = grad_JG_num/grad_JG_den
            return grad_JG
            
        def gradGZ():
            grad_GZ = self.a - torch.pow(self.a, 2)
            grad_GZ = grad_GZ.unsqueeze(0)
            return grad_GZ

        def gradZW():
            grad_ZW = self.inputVector
            return grad_ZW

        gradJWX = gradJW(gradJG(axis="x"), gradGZ(), gradZW())
        gradJWY = gradJW(gradJG(axis="y"), gradGZ(), gradZW())
        self.grad = torch.cat((gradJWX, gradJWY))

    def computeCost(self):
        self.cost = torch.sqrt(torch.sum((self.ballCoordinate.unsqueeze(0) - self.goalCoordinate.unsqueeze(0))**2, axis=1))

    def gradientDescent(self, learing_rate):
        self.weights = self.weights - learing_rate*self.grad

net = Network(4, 3, 3)
print("Weights")
print(net.weights)
net.computeInput()
print("Ball Coordinates")
print(net.ballCoordinate)
print("Node Coordinates")
print(net.nodeCoordinate)
print("Cost")
net.computeCost()
print(net.cost)
net.forwardProp()
net.updateBallCoord()
net.updateNodeCoord()
print("Ball Coordinates")
print(net.ballCoordinate)
print("Node Coordinates")
print(net.nodeCoordinate)
net.backProp()
net.gradientDescent(0.2)
print("Weights")
print(net.weights)
print("Cost")
net.computeCost()
print(net.cost)