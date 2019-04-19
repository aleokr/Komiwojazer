import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt
from appJar import gui
from threading import Thread
import time


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


def breedNWOX(parent1, parent2): #metoda poprawiona według artukułu
    
    rand1=int(random.random() * len(parent1))
    rand2 =int(random.random() * len(parent1))
    a=min(rand1, rand2)
    b=max(rand1,rand2)
    
    child1 = [i for i in range(len(parent1)+b-a)]
    child2 = [i for i in range(len(parent2)+b-a)]
    child1[0] = parent1[0]
    child2[0] = parent2[0]
    
    parent1Set = set()#zbiory rodziców z wylosowanego przedziału
    parent2Set = set()
    i=a
    while i<b:
        parent1Set.add(parent1[i])
        parent2Set.add(parent2[i])
        i+=1

    i=a#założenia metody
    while i<b:
        if (parent1[i] in parent2Set)!=True: 
            child1[i] = parent1[i]

        if (parent2[i] in parent1Set)!=True:
            child2[i] = parent2[i]
        i=i+1

    return child1,child2

def breedCX(parent1, parent2):#nietestowane
    child1 = [i for i in range(len(parent1))]
    child2 = [i for i in range(len(parent2))]
    child1[0] = parent1[0]
    child2[0] = parent2[0]
    i = 0
    while parent2[i] not in child1:
        j = parent1.index(parent2[i])
        child1[j] = parent1[j]
        child2[j] = parent2[j]
        i=j

    for gene in child1:
        if gene is None :
            child1[child1.index(gene)] = parent2[child1.index(gene)]

    for gene in child2:
        if gene is None :
            child2[child2.index(gene)] = parent1[child2.index(gene)]

    return child1,child2


def breedPMX(parent1, parent2):
    child1 = [i for i in range(len(parent1))]
    child2 = [i for i in range(len(parent2))]
    child1[0] = parent1[0]
    child2[0] = parent2[0]
    position1 = [i+1 for i in range(len(parent1))]
    position2 = [i+1 for i in range(len(parent2))]
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)
    tmp = 0

    for i in range(startGene, endGene):
        tmp1=child1[i]
        tmp2=child2[i]
        child1[i]=tmp2
        child1[position1[tmp1]]=tmp1
        child2[i]=tmp2
        child2[position2[tmp2]]=tmp2
        tmp = position1[tmp1]
        position1[tmp1]= position1[tmp2]
        position1[tmp2]= tmp
        tmp = position2[tmp1]
        position2[tmp1]= position2[tmp2]
        position2[tmp2]= tmp

    return child1,child2


def breedUMPX(parent1, parent2):
    child1 = parent1
    child2 = parent2
    position1 = [i+1 for i in range(len(parent1))]
    position2 = [i+1 for i in range(len(parent2))]
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(len(parent1)):
        q=random.random()
        p=0.5 
        if q>=p:
            tmp1=child1[i]
            tmp2=child2[i]
            child1[i]=tmp2
            child1[position1[tmp1]]=tmp1
            child2[i]=tmp2
            child2[position2[tmp2]]=tmp2
            position1[tmp1]= position1[tmp2]
            position1[tmp2]= position1[tmp1]
            position2[tmp1]= position2[tmp2]
            position2[tmp2]= position2[tmp1]

    return child1,child2



def breedOX(parent1, parent2):
    
    rand1=int(random.random() * len(parent1))
    rand2 =int(random.random() * len(parent1))
    a=min(rand1, rand2)
    b=max(rand1,rand2)
    j1=j2=k=b+1

    child1 = [i for i in range(len(parent1))]
    child2 = [i for i in range(len(parent2))]
    parent1Set = set()
    parent2Set = set()
    i=a
    while i<b:
        parent1Set.add(parent1[i])
        parent2Set.add(parent2[i])
        i+=1

    for i in range(len(parent1)):
        if (parent1[i] in parent1Set)!=True: 
            child1[j1] = parent1[k]
            j1+=1
        if (parent2[i] in parent2Set)!=True:
            child2[j2] = parent2[k]
            j2+=1
        k+=1
    
    return child1,child2


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
    geneticAlgorithm(population=cityList, popSize=50, eliteSize=5, mutationRate=0.01, generations=200,whichCrossover=4)
    geneticAlgorithmPlot(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500,whichCrossover=4)


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
