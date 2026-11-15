# [repro-patch P1] datasets module restored from upstream coperception (master).
# GCP README states the project builds on https://github.com/coperception/coperception
# and gcp.py imports `from coperception.datasets import V2XSimDet`, but the
# released GCP snapshot ships no datasets package. V2XSimDet.py and
# NuscenesDataset.py are copied verbatim from upstream coperception master
# (zip sha256 2a57e8270fe8dc302adcc0222ec6db7a1d62efba226574d386ca4ded80bfb2d6).
# V2XSimSeg is NOT re-exported here because only the detection loader is used
# by every GCP entry point (gcp.py / train_lstm_ae.py / train_codet.py).
from .V2XSimDet import V2XSimDet
from .NuscenesDataset import NuscenesDataset
