import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt
from appJar import gui
from threading import Thread
import time
import itertools
import math
import copy
import pylab 

class City: #miasto
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    
    def __repr__(self): #informacja o mieście
        return "(" + str(self.x) + "," + str(self.y) + ")"


class Fitness: #odwrotność długości całej trasy (dlaczego bierzemy odwrotność ????????)
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness= 0.0
    
    def routeDistance(self):
        if self.distance ==0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route): #jeśli iteracja nie jest większa od długości -> nie przeszedl przez wszystkie jeszcze
                    toCity = self.route[i + 1]
                else:#jeśli przeszedł przez wszystkie wraca na start
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity) #dodaje odległość miedzy odpowiednimi miastami 
            self.distance = pathDistance
        return self.distance
    
    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness


def rankRoutes(population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)#sortujemy wedlug najwiekszych i odwracamy ??????


def createRoute(cityList): #pojedynczy osobnik(trasa) (randomowo wybierana kolejność miast)
    route = random.sample(cityList, len(cityList))
    return route


def initialPopulation(popSize, cityList): #robi całą populację tras (o zadanej liczebności) ##potrzebne tylko do początkowej populacji (kreacjonizm mocno :D )
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population



def selection(popRanked, eliteSize): #tutaj wybieramy, którym osobnikom pozwolimy się rozmnażać (według Fitness proportionate selection - im większe Fitness, tym większe prawdopodobieństwo rozrodu)
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"]) #tworzymy sobie dwuwymiarową tabele
    df['cum_sum'] = df.Fitness.cumsum() 
    
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    
    for i in range(0, eliteSize): #eliteSize to liczebność osobników, które poradziły sobie najlepiej i dostają gwarancję rozrodu
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):#tutaj losujemy z reszty osobników
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults


def matingPool(population, selectionResults):# to jest funkcja do ekstrakcji osobników z licencją na rozmnażanie spośród całej populacji
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


#Mechanizm Cross-Over


def breedNWOX(parent1, parent2): 
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child




def breedPopulation(matingpool, eliteSize):#rozmnażator według testowego cross-over
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,eliteSize):
        children.append(matingpool[i])
    
    for i in range(0, length):
        child = breedNWOX(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children


#Mechanizmy mutacji

def mutate(individual, mutationRate):#z tutoriala - tutaj polega na randomowej zamianie miejscami dwóch miast (funkcja dla jednego osobnika) ##w artykule jest inaczej - losuje dwa miejsca w chromosomie i odwraca wszystko pomiędzy
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))
            
            city1 = individual[swapped]
            city2 = individual[swapWith]
            
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual

def mutateRSM(parent1, parent2): #zależnie od whichWay odpowiedni crossover 
    rand1=int(random.random() * len(parent1))
    rand2 =int(random.random() * len(parent1))
    a=min(rand1, rand2)
    b=max(rand1,rand2)
    while a<b:
        breedNWOX(parent1, parent2) 
        a+=1
        b-=1


def mutatePopulation(population, mutationRate): #funkcja mutująca całą populację
    mutatedPop = []
    
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop


def nextGeneration(currentGen, eliteSize, mutationRate): #wytworzenie nowej populacji
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration


def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    geneticList=[]
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        geneticList.append(1 / rankRoutes(pop)[0][1])  
    
    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    
    return geneticList


def geneticAlgorithmPlot(listGenetic, results): 
    global Genetic, Neighbour, Optimal,Graph
    progress = []
    optiProgress = []
    neighbourProgress = []
    for i in range(len(listGenetic)):
        progress.append(listGenetic[i])
        optiProgress.append(results[0])
        neighbourProgress.append(results[1])
    
    if Optimal==True:
        pylab.plot(optiProgress,label='Optimal Route')
    if Neighbour==True:
        pylab.plot(neighbourProgress,label='Nearest neighbour')
    if Genetic==True:
        pylab.plot(progress, label='Genetic algorithm')
    

    
    pylab.ylabel('Distance')
    pylab.xlabel('Generation')
    pylab.legend(loc='upper right')
    pylab.title("Cruel Salesman")
    if Graph==True:
        pylab.show()

def cost(route):
    sum = 0
    route.append(route[0])
    while len(route) > 1:
        p0, *route = route
        sum += math.sqrt((int(p0.x) - int(route[0].x))**2 + (int(p0.y) - int(route[0].y))**2)
    return sum

def optimalAlgorithm(route):
    d = float("inf")
    for p in itertools.permutations(route):
        c = cost(list(p))
        if c <= d:
            d = c
            pmin = p
    print("Optimal route:", pmin)
    print("Length:", d)
    return d

def closestpoint(point, route):
    dmin = float("inf")
    for p in route:
        d = math.sqrt((int(point.x) - int(p.x))**2 + (int(point.y) - int(p.y))**2)
        if d < dmin:
            dmin = d
            closest = p
    return closest, dmin

def nearestNeighbour(route):
    point, *route = route
    path = [point]
    sum = 0
    while len(route) >= 1:
        closest, dist = closestpoint(path[-1], route)
        path.append(closest)
        route.remove(closest)
        sum += dist
    closest, dist = closestpoint(path[-1], [point])
    path.append(closest)
    sum += dist
    print("Optimal route:", path)
    print("Length:", sum)
    return sum



def calculate():
    global Genetic, Neighbour, Optimal,Graph
    cityList = []
    for i in range(0,9):
        cityList.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))
    if Optimal==True:
        optiRoute = optimalAlgorithm(copy.deepcopy(cityList))
    if Neighbour==True:
        nearNeig = nearestNeighbour(copy.deepcopy(cityList))
    if Genetic==True:
        geneticList=geneticAlgorithm(population=copy.deepcopy(cityList), popSize=50, eliteSize=5, mutationRate=0.01, generations=200)
    if Graph==True:
        if Optimal==False:
            optiRoute=[]
        if Neighbour==False:
            nearNeig=[]
        if Genetic==False:
            geneticList=[]
        geneticAlgorithmPlot(geneticList,results=[optiRoute, nearNeig])


def press(button):
    if button == "Run":
        t = Thread(target=calculate)
        t.start()



#"main"
Genetic = True
Neighbour = False
Optimal = True
Graph = False

app = gui()

app.addLabel("title", "Get ready for the Cruel Salesman")
app.addLabel("Select cross-over algorithms:")
app.addButton("Run", press)


app.addCheckBox("Genetic algorithm")
app.addCheckBox("Optimal Route")
app.addCheckBox("Nearest Neighbour")
app.addCheckBox("Draw graph")

#app.setCheckBoxChangeFunction("Genetic algorithm",crossoverSelected)
#app.setCheckBoxChangeFunction("Optimal Route",crossoverSelected)
#app.setCheckBoxChangeFunction("Nearest Neighbour",crossoverSelected))

app.go()
