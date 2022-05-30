from operator import attrgetter
from sklearn.covariance import LedoitWolf
import numpy.random as np
import numpy

lw = LedoitWolf()


class EDA(object):
    def __init__(self, init_population, mu, lambda_):
        self.population = init_population

        lw.fit(self.population)

        self.location_ = lw.location_
        self.dim = len(lw.location_)
        self.cov_ = lw.covariance_
        self.lambda_ = lambda_
        self.mu = mu

    def generate(self, ind_init):
        # Generate lambda_ individuals and put them into the provided class
        nrg = np.default_rng()
        arz = [nrg.multivariate_normal(self.location_, self.cov_) for i in range(self.lambda_)]  # duda!!!!!!!!!!1

        return list(map(ind_init, arz))

    def update(self, population):
        # Sort individuals so the best is first
        sorted_pop = sorted(population, key=attrgetter("fitness"), reverse=True)

        # Compute the average of the mu best individuals
        new_pop = sorted_pop[:self.mu]  # - self.centroid
        # avg = numpy.mean(z, axis=0)
        #
        # # Adjust variances of the distribution
        # self.sigma = numpy.sqrt(numpy.sum(numpy.sum((z - avg) ** 2, axis=1)) / (self.mu * self.dim))
        # self.centroid = self.centroid + avg
        lw.fit(new_pop)
        self.cov_ = lw.covariance_
        self.location_ = lw.location_
