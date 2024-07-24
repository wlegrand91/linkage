import linkage
from linkage.experiment import Experiment
from linkage.organizer import GlobalModel


# Make a list of all classes in linkage.models
available_models = []
for _k in linkage.models.__dict__:
    if _k.startswith("_"):
        continue

    if issubclass(type(linkage.models.__dict__[_k]),type):
        available_models.append(_k)