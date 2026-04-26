from .predictor import Predictor
from .predictor_pix2poly import Pix2PolyPredictor

try:
    from .predictor_ffl import FFLPredictor
except ImportError:
    pass

try:
    from .predictor_hisup import HiSupPredictor
except ImportError:
    pass