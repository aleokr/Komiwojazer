import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt
from appJar import gui
from threading import Thread
import time
#from __future__ import division
import warnings
#from collections import Sequence
#from itertools import repeat

class City: #miasto
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    
    def distance(self, City):
        xDis = abs(self.x - City.x)
        yDis = abs(self.y - City.y)
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
    df['cum_sum'] = df.Fitness.cumsum() #???
    
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


#Mechanizmy Cross-Over


def breedNWOX(ind1, ind2):
    """Executes a blend crossover that modify in-place the input individuals.
    The blend crossover expects :term:`sequence` individuals of floating point
    numbers.
    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param alpha: Extent of the interval in which the new values can be drawn
                  for each attribute on both side of the parents' attributes.
    :returns: A tuple of two individuals.
    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    alpha=2
    for i, (x1, x2) in enumerate(zip(ind1, ind2)):
        gamma = (1. + 2. * alpha) * random.random() - alpha
        ind1[i] = (1. - gamma) * x1 + gamma * x2
        ind2[i] = gamma * x1 + (1. - gamma) * x2

    return ind1, ind2

def breedCX(ind1, ind2):#messy crossover
    """Executes a one point crossover on :term:`sequence` individual.
    The crossover will in most cases change the individuals size. The two
    individuals are modified in place.
    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.
    This function uses the :func:`~random.randint` function from the python base
    :mod:`random` module.
    """
    cxpoint1 = random.randint(0, len(ind1))
    cxpoint2 = random.randint(0, len(ind2))
    ind1[cxpoint1:], ind2[cxpoint2:] = ind2[cxpoint2:], ind1[cxpoint1:]

    return ind1, ind2


def breedPMX(ind1, ind2):
    """Executes a partially matched crossover (PMX) on the input individuals.
    The two individuals are modified in place. This crossover expects
    :term:`sequence` individuals of indices, the result for any other type of
    individuals is unpredictable.
    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.
    Moreover, this crossover generates two children by matching
    pairs of values in a certain range of the two parents and swapping the values
    of those indexes. For more details see [Goldberg1985]_.
    This function uses the :func:`~random.randint` function from the python base
    :mod:`random` module.
    .. [Goldberg1985] Goldberg and Lingel, "Alleles, loci, and the traveling
       salesman problem", 1985.
    """
    size = min(len(ind1), len(ind2))
    p1, p2 = [0]*size, [0]*size

    # Initialize the position of each indices in the individuals
    for i in xrange(size):
        p1[ind1[i]] = i
        p2[ind2[i]] = i
    # Choose crossover points
    cxpoint1 = random.randint(0, size)
    cxpoint2 = random.randint(0, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    # Apply crossover between cx points
    for i in xrange(cxpoint1, cxpoint2):
        # Keep track of the selected values
        temp1 = ind1[i]
        temp2 = ind2[i]
        # Swap the matched value
        ind1[i], ind1[p1[temp2]] = temp2, temp1
        ind2[i], ind2[p2[temp1]] = temp1, temp2
        # Position bookkeeping
        p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
        p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

    return ind1, ind2


def breedUMPX(ind1, ind2):
    """Executes a uniform partially matched crossover (UPMX) on the input
    individuals. The two individuals are modified in place. This crossover
    expects :term:`sequence` individuals of indices, the result for any other
    type of individuals is unpredictable.
    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.
    Moreover, this crossover generates two children by matching
    pairs of values chosen at random with a probability of *indpb* in the two
    parents and swapping the values of those indexes. For more details see
    [Cicirello2000]_.
    This function uses the :func:`~random.random` and :func:`~random.randint`
    functions from the python base :mod:`random` module.
    .. [Cicirello2000] Cicirello and Smith, "Modeling GA performance for
       control parameter optimization", 2000.
    """
    size = min(len(ind1), len(ind2))
    p1, p2 = [0]*size, [0]*size

    # Initialize the position of each indices in the individuals
    for i in range(size):
        p1[ind1[i]] = i
        p2[ind2[i]] = i

    for i in range(size):
        if random.random() < 0.5:
            # Keep track of the selected values
            temp1 = ind1[i]
            temp2 = ind2[i]
            # Swap the matched value
            ind1[i], ind1[p1[temp2]] = temp2, temp1
            ind2[i], ind2[p2[temp1]] = temp1, temp2
            # Position bookkeeping
            p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
            p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

    return ind1, ind2



def breedOX(ind1, ind2):
    """Executes an ordered crossover (OX) on the input
    individuals. The two individuals are modified in place. This crossover
    expects :term:`sequence` individuals of indices, the result for any other
    type of individuals is unpredictable.
    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.
    Moreover, this crossover generates holes in the input
    individuals. A hole is created when an attribute of an individual is
    between the two crossover points of the other individual. Then it rotates
    the element so that all holes are between the crossover points and fills
    them with the removed elements in order. For more details see
    [Goldberg1989]_.
    This function uses the :func:`~random.sample` function from the python base
    :mod:`random` module.
    .. [Goldberg1989] Goldberg. Genetic algorithms in search,
       optimization and machine learning. Addison Wesley, 1989
    """
    size = min(len(ind1), len(ind2))
    a, b = random.sample(range(size), 2)
    if a > b:
        a, b = b, a

    holes1, holes2 = True*size, True*size
    for i in range(size):
        if i < a or i > b:
            holes1[ind2[i]] = False
            holes2[ind1[i]] = False

    # We must keep the original values somewhere before scrambling everything
    temp1, temp2 = ind1, ind2
    k1 , k2 = b + 1, b + 1
    for i in range(size):
        if not holes1[temp1[(i + b + 1) % size]]:
            ind1[k1 % size] = temp1[(i + b + 1) % size]
            k1 += 1

        if not holes2[temp2[(i + b + 1) % size]]:
            ind2[k2 % size] = temp2[(i + b + 1) % size]
            k2 += 1

    # Swap the content between a and b (included)
    for i in range(a, b + 1):
        ind1[i], ind2[i] = ind2[i], ind1[i]

    return ind1, ind2


def breedPopulation(matingpool, eliteSize,whichCrossover):#rozmnażator według testowego cross-over
    #numery crossovery według kolejności: 1-NWOX 2-CX 3-PMX 4-UMPX 5-OX 
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,eliteSize):
        children.append(matingpool[i])
    
    for i in range(0, length):
        if(whichCrossover==1):
            child = breedNWOX(pool[i], pool[len(matingpool)-i-1])
        elif(whichCrossover==2):
            child = breedCX(pool[i], pool[len(matingpool)-i-1])
        elif(whichCrossover==3):
            child = breedPMX(pool[i], pool[len(matingpool)-i-1])
        elif(whichCrossover==4):
            child = breedUMPX(pool[i], pool[len(matingpool)-i-1])
        elif(whichCrossover==5):
            child = breedOX(pool[i], pool[len(matingpool)-i-1])
        children.append(child[0])
        children.append(child[1])
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


def mutateRSM(individual):#individual - osobnik do mutacji
    rand1=int(random.random() * len(individual))
    rand2 =int(random.random() * len(individual))
    a=min(rand1, rand2)
    b=max(rand1,rand2)
    while a<b:
       tmp = individual[a]
       individual[a] = individual[b]
       individual[b] = tmp
       a = a+1
       b = b-1
    return individual


def mutatePopulation(population, mutationRate): #funkcja mutująca całą populację
    mutatedPop = []
    
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop


def nextGeneration(currentGen, eliteSize, mutationRate,whichCrossover): #wytworzenie nowej populacji
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize, whichCrossover) 
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration


def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations,whichCrossover):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))
    
    for i in range(0, generations):#założyłam że dla kilku wyborów pętla będzie już w mainie 
        pop = nextGeneration(pop, eliteSize, mutationRate,whichCrossover)
    
    print("Final distance for crossover"+ whichCrossover+ " and mutation "+mutationRate+": " + str(1 / rankRoutes(pop)[0][1]))#to dla wszystkich
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute


def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations,whichCrossover): #pokazuje wykres najlepszej drogi dla danego pokolenia
    pop = initialPopulation(popSize, population)
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate,whichCrossover)
        progress.append(1 / rankRoutes(pop)[0][1])
    
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()



def calculate():
    cityList = []
    for i in range(0,25):
        cityList.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))
    geneticAlgorithm(population=cityList, popSize=50, eliteSize=5, mutationRate=0.01, generations=200,whichCrossover=5)
    geneticAlgorithmPlot(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500,whichCrossover=5)


def press(button):
    if button == "Run":
        t = Thread(target=calculate)
        t.start()

def crossoverSelected(checkBox):
    global CX, PMX, UPMX, NWOX, OX
    if(checkBox == "Cycle crossover"):
        CX = not CX
    elif (checkBox == "Partially-Mapped Crossover"):
        PMX = not PMX
    elif (checkBox == "Uniform PMX"):
        UPMX = not UPMX
    elif (checkBox == "Non-wrapping ordered crossover"):
        NWOX = not NWOX
    elif (checkBox == "Ordered crossover"):
        OX = not OX


def mutationSelected(radioButton):
    global pairMutation, groupMutation
    if radioButton == "Pair mutation":
        pairMutation = True
        groupMutation = False
    else:
        pairMutation = False
        groupMutation = True
  

#"main"
CX = False
PMX = False
UPMX = False
NWOX = False
OX = False
pairMutation = True
groupMutation = False

app = gui()

app.addLabel("title", "Get ready for the Cruel Salesman")
app.addLabel("Select cross-over algorithms:")
app.addButton("Run", press)

app.addCheckBox("Cycle crossover")
app.addCheckBox("Partially-Mapped Crossover")
app.addCheckBox("Uniform PMX")
app.addCheckBox("Non-wrapping ordered crossover")
app.addCheckBox("Ordered crossover")
app.addLabel("Select mutation mechanism: ")
app.addRadioButton("mutation", "Pair mutation")
app.addRadioButton("mutation", "Group mutation")
app.setRadioButtonChangeFunction("mutation", mutationSelected)

app.setRadioButton("mutation","Pair mutation" )

app.setCheckBoxChangeFunction("Cycle crossover",crossoverSelected)
app.setCheckBoxChangeFunction("Partially-Mapped Crossover",crossoverSelected)
app.setCheckBoxChangeFunction("Uniform PMX",crossoverSelected)
app.setCheckBoxChangeFunction("Non-wrapping ordered crossover",crossoverSelected)
app.setCheckBoxChangeFunction("Ordered crossover",crossoverSelected)

app.go()
