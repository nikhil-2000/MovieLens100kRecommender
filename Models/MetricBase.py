




class MetricBase:

    def __init__(self):
        self.hits = 0
        self.ranks = []
        self.mrr_ranks = []
        self.average_precisions = []
        self.recall = []

    def mean_reciprocal_rank(self):
        return sum(self.mrr_ranks) / len(self.mrr_ranks)

    def get_average_precision(self):
        return sum(self.average_precisions) / len(self.average_precisions)

    def get_recall(self):
        return sum(self.recall) / len(self.recall)

    def hitrate(self, tests):
        return 100 * self.hits / tests

    def mean_rank(self):
        return sum(self.ranks) / len(self.ranks)