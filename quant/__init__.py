from .lsq import LSQQuantStrategy
from .pact import PACTActivationQuantizer
from .adaround import AdaRoundQuantStrategy
from .apot import APoTQuantizer
from .qdrop import QDropQuantizer


QUANTIZER_MAP = {
    "lsq": LSQQuantStrategy,
    "pact": PACTActivationQuantizer,
    "apot": APoTQuantizer,
    "qdrop": QDropQuantizer,
    "adaround": AdaRoundQuantStrategy,
}
