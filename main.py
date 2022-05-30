# Estimation of Distribution Algorithm
import numpy
from deap import base, creator, tools, benchmarks, algorithms
from deap.tools import HallOfFame

from indGen import IndGen
from EDA import EDA

creator.create("MyFitness", base.Fitness, weights=(-1.0,))
creator.create("Individual", numpy.ndarray, fitness=creator.MyFitness)

toolbox = base.Toolbox()

# INPUTS:
# A representation of the solutions

# An objective function f
objFun = benchmarks.schaffer
toolbox.register("evaluate", objFun)

# P0 <- Generate initial population according to the given representation
i_gen = IndGen()

toolbox.register("individual", tools.initIterate, creator.Individual, i_gen.ind_Gen)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)
init_pop = toolbox.population(n=1000)


# F0 <- Evaluate individuals of P0 using f
# g <- 1
# while termination criteria are not met do
#   Sg <- Select a subset of Pg−1 according to Fg−1 using a selection mechanism
#   Pg(x) <- Estimate the probability of solutions in Sg
#   Qg <- Sample Sg(x) according to the given representation
#   Hg <- Evaluate individuals of Qg using f
#   Pg <- Incorporate Qg into Pg−1 according to Fg−1 and Hg
#   Fg <- Update Fg−1 according to the solutions in Pg
#   g <- g+1
# end while
# Output: The best solution(s) in Pg−1
# noinspection PyUnresolvedReferences
def main():
    LAMBDA = 1000
    MU = int(LAMBDA / 25)
    strategy = EDA(init_pop, mu=MU, lambda_=LAMBDA)

    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    # Numpy equality function (operators.eq) between two arrays returns the
    # equality element wise, which raises an exception in the if similar()
    # check of the hall of fame. Using a different equality function like
    # numpy.array_equal or numpy.all close solve this issue.
    hof: HallOfFame = tools.HallOfFame(1, similar=numpy.array_equal)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    algorithms.eaGenerateUpdate(toolbox, ngen=150, stats=stats, halloffame=hof)

    print(hof[0], " ", hof[0].fitness.values[0])

    return hof[0].fitness.values[0]


if __name__ == "__main__":
    main()
