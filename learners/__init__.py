from .q_learner import QLearner
from .q_learner_divide import QDivedeLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["q_divide_learner"] = QDivedeLearner
