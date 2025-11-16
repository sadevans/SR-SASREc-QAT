from .lsq import LSQQuantizer
from .pact import PACTQuantizer
from .adaround import AdaRoundQuantizer
from .apot import APoTQuantizer
from .qdrop import QDropQuantizer


QUANTIZER_MAP = {
    "lsq": LSQQuantizer,
    "pact": PACTQuantizer,
    "apot": APoTQuantizer,
    "qdrop": QDropQuantizer,
    "adaround": AdaRoundQuantizer,
}
