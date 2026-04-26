import hydra
import pandas as pd

from pixelspointspolygons.eval import Evaluator
from pixelspointspolygons.predict import Pix2PolyPredictor
from pixelspointspolygons.misc.shared_utils import setup_ddp, setup_hydraconf

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):
    
    setup_hydraconf(cfg)
    local_rank, world_size = setup_ddp(cfg)
    
    print(f"Predict {cfg.experiment.model.name}/{cfg.experiment.name} on {cfg.experiment.dataset.country}/{cfg.evaluation.split}")
    
    if cfg.experiment.model.name == "ffl":
        # We don't use FFL, so if it's somehow called, raise an error
        raise NotImplementedError("FFL model not available in this environment")
    elif cfg.experiment.model.name == "hisup":
        raise NotImplementedError("HiSup model not available in this environment")
    elif cfg.experiment.model.name == "pix2poly":
        predictor = Pix2PolyPredictor(cfg, local_rank, world_size)
    else:
        raise ValueError(f"Unknown model name: {cfg.experiment.model.name}")
    
    predictor.predict_dataset(split=cfg.evaluation.split)
    
    print(f"Evaluate {cfg.experiment.model.name}/{cfg.experiment.name} on {cfg.experiment.dataset.country}/{cfg.evaluation.split}")

    ee = Evaluator(cfg)
    ee.pbar_disable = False
    ee.load_gt(cfg.experiment.dataset.annotations[cfg.evaluation.split])
    ee.load_predictions(cfg.evaluation.pred_file)
    res = ee.evaluate()
    
    df = pd.DataFrame.from_dict(res, orient='index')
    print("\n", df, "\n")
    print(f"Save eval file to {cfg.evaluation.eval_file}")
    df.to_csv(cfg.evaluation.eval_file, index=True, float_format="%.3g")
        
if __name__ == "__main__":
    main()