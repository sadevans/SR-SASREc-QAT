from .lsq import LSQQuantStrategy
from .pact import PACTQuantStrategy
from .adaround import AdaRoundQuantStrategy
from .apot import APoTQuantStrategy
from .qdrop import QDropQuantStrategy


QUANTIZER_MAP = {
    "lsq": LSQQuantStrategy,
    "pact": PACTQuantStrategy,
    "apot": APoTQuantStrategy,
    "qdrop": QDropQuantStrategy,
    "adaround": AdaRoundQuantStrategy,
}
