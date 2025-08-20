from NeuralNetUtil import buildExamplesFromCarData,buildExamplesFromPenData, buildExamplesFromExtraData
from NeuralNet import buildNeuralNet
from math import pow, sqrt
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
computer_hardware = fetch_ucirepo(id=29) 
  
# data (as pandas dataframes) 
X = computer_hardware.data.features 
y = computer_hardware.data.targets 
  
# metadata 
print(computer_hardware.metadata) 
  
# variable information 
print(computer_hardware.variables) 

def average(argList):
    return sum(argList) / float(len(argList))
    
def stDeviation(argList):
    mean = average(argList)
    diffSq = [pow((val-mean),2) for val in argList]
    return sqrt(sum(diffSq)/len(argList))

penData = buildExamplesFromPenData()
def testPenData(hiddenLayers = [24]):
    return buildNeuralNet(penData, maxItr = 200, hiddenLayerList = hiddenLayers)

carData = buildExamplesFromCarData()
def testCarData(hiddenLayers = [16]):
    return buildNeuralNet(carData, maxItr = 200,hiddenLayerList = hiddenLayers)

def question5():
    i = 0
    accuraciesPenData = []
    while i < 5:
        accuraciesPenData.append(testPenData()[1])
        i = i + 1

    i = 0
    accuraciesCarData = []
    while i < 5:
        accuraciesCarData.append(testCarData()[1])
        i = i + 1

    print(accuraciesPenData)
    print('Pen Data Max: %f\nPen Data Standard Deviation: %f\nPen Data Average: %f'%(max(accuraciesPenData), stDeviation(accuraciesPenData), average(accuraciesPenData)))
    print(accuraciesCarData)
    print('Car Data Max: %f\nCar Data Standard Deviation: %f\nCar Data Average: %f'%(max(accuraciesCarData), stDeviation(accuraciesCarData), average(accuraciesCarData)))
# question5()

def question6():
    increments = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    
    penDataResults = {}
    carDataResults = {}
    
    for inc in increments:
        penAccuracies = []
        carAccuracies = []
        
        # Run the tests 5 times for each increment
        for _ in range(5):
            penAccuracies.append(testPenData([inc])[1])
            carAccuracies.append(testCarData([inc])[1])
        
        penDataResults[inc] = {
            'max': max(penAccuracies),
            'average': average(penAccuracies),
            'std_dev': stDeviation(penAccuracies)
        }
        
        carDataResults[inc] = {
            'max': max(carAccuracies),
            'average': average(carAccuracies),
            'std_dev': stDeviation(carAccuracies)
        }
    
    # Print the results in a table format
    print("Pen Data Results:")
    print("Hidden Layer Size | Max Accuracy | Average Accuracy | Standard Deviation")
    for inc in increments:
        print(f"{inc:<18} | {penDataResults[inc]['max']:<12.6f} | {penDataResults[inc]['average']:<16.6f} | {penDataResults[inc]['std_dev']:<18.6f}")
    
    print("\nCar Data Results:")
    print("Hidden Layer Size | Max Accuracy | Average Accuracy | Standard Deviation")
    for inc in increments:
        print(f"{inc:<18} | {carDataResults[inc]['max']:<12.6f} | {carDataResults[inc]['average']:<16.6f} | {carDataResults[inc]['std_dev']:<18.6f}")
    
    # Produce a learning curve
    penAverageAccuracies = [penDataResults[inc]['average'] for inc in increments]
    carAverageAccuracies = [carDataResults[inc]['average'] for inc in increments]
# question6()

def getAndBuildXORData(fileString="datasets/xor.txt", limit=5):
    examples = []
    data = open(fileString)
    lineNum = 0
    for line in data:
        if lineNum >= limit:
            break

        parts = line.strip().split()

        if len(parts) != 3:
            continue

        input1, input2, output = map(int, parts)

        examples.append(([input1, input2], [output]))
        lineNum += 1
    data.close()
    return examples

# xorDataTrain = getAndBuildXORData("datasets/xor.txt",5)
# xorDataTest = getAndBuildXORData("datasets/xor.txt",5)
# xorData = [xorDataTrain, xorDataTest]

def testXORData(hiddenLayers = [100]):
    return buildNeuralNet(xorData, maxItr = 200, hiddenLayerList = hiddenLayers)

def question7():
    i = 0
    accuraciesXORData = []
    while i < 5:
        accuraciesXORData.append(testXORData()[1])
        i = i + 1

    print(accuraciesXORData)
    print('XOR Data Max: %f\nXOR Data Standard Deviation: %f\nXOR Data Average: %f'%(max(accuraciesXORData), stDeviation(accuraciesXORData), average(accuraciesXORData)))
# question7()

# extraData = buildExamplesFromExtraData()

def testExtraData(hiddenLayers = [30]):
    return buildNeuralNet(extraData, maxItr = 200, hiddenLayerList = hiddenLayers)

def question8():
    i = 0
    accuraciesExtraData = []
    while i < 5:
        accuraciesExtraData.append(testExtraData()[1])
        i = i + 1

    print(accuraciesExtraData)
    print('Extra Data Max: %f\nExtra Data Standard Deviation: %f\nExtra Data Average: %f'%(max(accuraciesExtraData), stDeviation(accuraciesExtraData), average(accuraciesExtraData)))

# question8()