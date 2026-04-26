from .trainer_pix2poly import Pix2PolyTrainer

try:
    from .trainer import Trainer
except Exception:
    pass

try:
    from .trainer_hisup import HiSupTrainer
except Exception:
    pass

try:
    from .trainer_ffl import FFLTrainer
except Exception:
    pass
