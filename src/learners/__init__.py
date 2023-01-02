from .cq_learner import CQLearner
from .cq_learner_divide import CQLearnerDivide
from .facmac_learner import FACMACLearner
from .facmac_learner_discrete import FACMACDiscreteLearner
from .maddpg_learner import MADDPGLearner
from .maddpg_learner_discrete import MADDPGDiscreteLearner

REGISTRY = {}
REGISTRY["cq_learner"] = CQLearner
REGISTRY["cq_learner_divide"] = CQLearnerDivide
REGISTRY["facmac_learner"] = FACMACLearner
REGISTRY["facmac_learner_discrete"] = FACMACDiscreteLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["maddpg_learner_discrete"] = MADDPGDiscreteLearner
