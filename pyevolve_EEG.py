from pyevolve import G1DBinaryString
from pyevolve import G1DList
from pyevolve import GSimpleGA
from pyevolve import Selectors
from pyevolve import Mutators
from MLPClassify_EEG import MLPClassify

# This function is the evaluation function, we want
# to give high score to more zero'ed chromosomes

mlpclassify = MLPClassify()

def eval_func(chromosome):
    score=mlpclassify.classify(chromosome)    
    return score
# Genome instance
genome = G1DBinaryString.G1DBinaryString(13)
# genome = G1DList.G1DList(10)
# genome.setParams(rangemin=0, rangemax=200)


# The evaluator function (objective function)
genome.evaluator.set(eval_func)
genome.mutator.set(Mutators.G1DBinaryStringMutatorFlip)

# Genetic Algorithm Instance
ga = GSimpleGA.GSimpleGA(genome)
ga.selector.set(Selectors.GTournamentSelector)
# ga.terminationCriteria.set(GSimpleGA.ConvergenceCriteria)
ga.setGenerations(100)
ga.setPopulationSize(2)
# ga.setMutationRate(0.05)

# Do the evolution, with stats dump 
# frequency of 10 generations
ga.evolve(freq_stats=10)

# Best individual
best = ga.bestIndividual()
print best
print best.score
print ga.getPopulation()
