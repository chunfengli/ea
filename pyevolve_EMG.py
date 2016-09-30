from pyevolve import G1DBinaryString
from pyevolve import GSimpleGA
from pyevolve import Selectors
from pyevolve import Mutators
from MLPClassify_EMG2 import MLPClassify

# This function is the evaluation function, we want
# to give high score to more zero'ed chromosomes

mlpclassify = MLPClassify()

def eval_func(chromosome):
    score=mlpclassify.classify(chromosome)    
    return score

# Genome instance
genome = G1DBinaryString.G1DBinaryString(60)

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
