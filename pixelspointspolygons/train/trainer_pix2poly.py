# disable annoying transfromers and albumentation warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import os
import torch
import wandb
import json
import shutil
import torch
import math
import csv

import torch.distributed as dist

from collections import defaultdict
from torch import nn
from torch import optim
from transformers import get_linear_schedule_with_warmup

from ..models.pix2poly import Tokenizer, Pix2PolyModel
from ..misc import AverageMeter, get_lr, get_tile_names_from_dataloader, denormalize_image_for_visualization
from ..predict.predictor_pix2poly import Pix2PolyPredictor as Predictor
from ..eval import Evaluator
from ..misc.debug_visualisations import *
from ..misc.coco_conversions import coco_anns_to_shapely_polys, tensor_to_shapely_polys

from .trainer import Trainer

# === OPTIM: cuDNN benchmark + SDP backend (étape 1) ===
# benchmark=True : cuDNN choisit automatiquement l'algo de convolution le plus rapide
# pour les tailles d'entrée fixes (224×224). deterministic=False lève la contrainte de
# reproductibilité exacte qui bride les algos non-déterministes mais plus rapides.
# Le backend SDP est sélectionné selon la génération du GPU :
#   ≥ sm_80 (A100/H100) → Flash Attention (flash_sdp)
#   < sm_80 (V100, etc.) → math SDP (plus stable sur Volta)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
_gpu_cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
if _gpu_cap >= (8, 0):
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
else:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


class Pix2PolyTrainer(Trainer):
        
    def setup_model(self):
        self.tokenizer = Tokenizer(self.cfg)
        self.model = Pix2PolyModel(self.cfg,self.tokenizer.vocab_size,local_rank=self.local_rank)
        
    
    def setup_optimizer(self):
        cfg = self.cfg.experiment.model
        grad_accum = getattr(cfg, "gradient_accumulation_steps", 1)

        # --- Optimizer (peak LR is cfg.learning_rate) ---
        # === OPTIM: fused AdamW (étape 1) ===
        # fused=True exécute toutes les mises à jour de paramètres en un seul kernel CUDA
        # au lieu d'un kernel par paramètre → ~10 % de gain sur les étapes d'optimisation.
        # Requiert CUDA ≥ sm_70 (V100+) et aucune couche LazyLinear (vérifié : absent ici).
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95),
            fused=True,
        )

        # --- Steps per epoch (prefer loader length to respect sampler/drop_last) ---
        if hasattr(self, "train_loader") and hasattr(self.train_loader, "__len__"):
            # len(train_loader) == number of BATCHES after sampler & drop_last
            steps_per_epoch = len(self.train_loader)
        else:
            # Fallback: compute from dataset size
            effective_batch = cfg.batch_size
            steps_per_epoch = math.ceil(len(self.train_loader.dataset) / effective_batch)

        # Account for gradient accumulation (optimizer steps per epoch)
        steps_per_epoch = max(1, math.ceil(steps_per_epoch / grad_accum))

        num_training_steps = cfg.num_epochs * steps_per_epoch
        num_warmup_steps = int(0.05 * num_training_steps)

        self.logger.info(
            f"Dataset={len(self.train_loader.dataset)}, "
            f"batch_size={cfg.batch_size}, grad_accum={grad_accum}, "
            f"steps/epoch={steps_per_epoch}, epochs={cfg.num_epochs}, "
            f"total_optimizer_steps={num_training_steps}, warmup_steps={num_warmup_steps}"
        )

        # --- Scheduler: warmup → linear decay to 0 ---
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        
    # def setup_loss_fn_dict(self):
        
    #     # Init loss functions
    #     weight = torch.ones(self.cfg.experiment.model.tokenizer.pad_idx + 1, device=self.cfg.host.device)
    #     weight[self.tokenizer.num_bins:self.tokenizer.BOS_code] = 0.0
    #     self.loss_fn_dict["coords"] = nn.CrossEntropyLoss(ignore_index=self.cfg.experiment.model.tokenizer.pad_idx, label_smoothing=self.cfg.experiment.model.label_smoothing, weight=weight)
    #     self.loss_fn_dict["perm"] = nn.BCELoss()
    
    def setup_loss_fn_dict(self):
        
        # Init loss functions
        
        self.loss_fn_dict["coords"] = nn.CrossEntropyLoss(ignore_index=self.cfg.experiment.model.tokenizer.pad_idx)
                
        self.loss_fn_dict["perm"] = nn.BCELoss()
    
    
    def visualization(self,  loader, epoch, predictor, coco_anns=None, num_images=2):
        
        self.model.eval()
        
        x_image, x_lidar, y_sequence, y_perm, tile_ids = next(iter(loader))
                
        if self.cfg.experiment.encoder.use_images:
            x_image = x_image.to(self.cfg.host.device, non_blocking=True)
            x_image = x_image[:num_images]
        if self.cfg.experiment.encoder.use_lidar:
            x_lidar = x_lidar.to(self.cfg.host.device, non_blocking=True)
            x_lidar = x_lidar.unbind()
            x_lidar = list(x_lidar)[:num_images]
            x_lidar = torch.nested.nested_tensor(x_lidar, layout=torch.jagged)
        
        split = loader.dataset.split
        outpath = os.path.join(self.cfg.output_dir,"visualizations", split)
        os.makedirs(outpath, exist_ok=True)
        self.logger.info(f"Save visualizations to {outpath}")
        
        if coco_anns is not None:
            coco_anns_dict = defaultdict(list)
            for ann in coco_anns:
                if ann["image_id"] >= num_images: 
                    break
                coco_anns_dict[ann["image_id"]].append(ann)
        
        if predictor is not None:
            predicted_polygons = predictor.batch_to_polygons(x_image, x_lidar, self.model, self.tokenizer)
            gt_polygons = predictor.coord_and_perm_to_polygons(y_sequence, y_perm)
            
            
        if self.cfg.experiment.encoder.use_lidar:
            lidar_batches = torch.unbind(x_lidar, dim=0)
            
        names = get_tile_names_from_dataloader(loader, tile_ids.cpu().numpy().flatten().tolist())

        for i in range(num_images):
        
            fig, ax = plt.subplots(1,2,figsize=(8, 4), dpi=150)
            ax = ax.flatten()

            if self.cfg.experiment.encoder.use_images:
                image = denormalize_image_for_visualization(x_image[i], self.cfg)
                plot_image(image, ax=ax[0])
                plot_image(image, ax=ax[1])
            if self.cfg.experiment.encoder.use_lidar:
                plot_point_cloud(lidar_batches[i], ax=ax[0])
                plot_point_cloud(lidar_batches[i], ax=ax[1])
            
            if coco_anns is not None:
                coco_polys = coco_anns_to_shapely_polys(coco_anns_dict[tile_ids[i].item()])
            else:
                coco_polys = []
            
            if predictor is not None:
                pred_polys = tensor_to_shapely_polys(predicted_polygons[i])
                gt_polys = tensor_to_shapely_polys(gt_polygons[i])
            else:
                pred_polys = []
                gt_polys = []
                
            if len(gt_polys):
                plot_shapely_polygons(gt_polys, ax=ax[0])
                
            if len(pred_polys) and not len(coco_polys):
                self.logger.debug("Plot predictions from loader")
                plot_shapely_polygons(pred_polys, ax=ax[1])
                
            if len(coco_polys):
                self.logger.debug("Plot coco predictions")
                plot_shapely_polygons(coco_polys, ax=ax[1])
                
            ax[0].set_title(f"GT_{split}_"+names[i])
            ax[1].set_title(f"PRED_{split}_"+names[i])
            
            plt.tight_layout()
            width = len(str(self.cfg.experiment.model.num_epochs))
            outfile = os.path.join(outpath, f"{epoch:0{width}d}_{names[i]}.png")
            self.logger.debug(f"Save visualization to {outfile}")
            plt.savefig(outfile)
            if self.cfg.run_type.log_to_wandb and self.local_rank == 0:
                wandb.log({f"{epoch:0{width}d}: {split}_{names[i]}": wandb.Image(fig)})            
            plt.close(fig)
        
    def split_coord_and_valence(self, y_sequence):
        """
        y_sequence: 
            [B, L] LongTensor   or
            [B, L, D] LongTensor
        returns:
            coords: [B, L, 2] LongTensor  (if input was 2D)
                    [B, L, 2, D] LongTensor (if input was 3D)
            valences: [B, L] LongTensor   (if input was 2D)
                    [B, L, D] LongTensor (if input was 3D)
        """
        if y_sequence.dim() == 2:  # [B, L]
            last_token = y_sequence[:, -1]
            y_sequence = y_sequence[:, :-1]
            
            B, L = y_sequence.shape
            grouped = y_sequence.view(B, L // 3, 3)
            
            coords = grouped[:, :, :2].reshape(B, -1)
            valences = grouped[:, :, 2]
            
            coords = torch.cat([coords, last_token.unsqueeze(1)], dim=1)
            valences = torch.cat([valences, last_token.unsqueeze(1)], dim=1)

        elif y_sequence.dim() == 3:  # [B, L, D]
            last_token = y_sequence[:, -1, :]  # [B, D]
            y_sequence = y_sequence[:, :-1, :]  # [B, L-1, D]
            
            B, L, D = y_sequence.shape
            grouped = y_sequence.view(B, L // 3, 3, D)
            
            coords = grouped[:, :, :2, :].reshape(B, -1, D)   # [B, 2*(L//3), D]
            valences = grouped[:, :, 2, :]                    # [B, L//3, D]
            
            coords = torch.cat([coords, last_token.unsqueeze(1)], dim=1)
            valences = torch.cat([valences, last_token.unsqueeze(1)], dim=1)

        else:
            raise ValueError("y_sequence must be 2D or 3D")

        return coords, valences

        
    def val_one_epoch(self):

        self.logger.info("Validate...")
        self.model.eval()
        
        for loss in self.loss_fn_dict.values():
            loss.eval()

        loss_meter = AverageMeter()
        coord_loss_meter = AverageMeter()
        perm_loss_meter = AverageMeter()

        loader = self.progress_bar(self.val_loader)
        
        lidar_dropout = None if not hasattr(self.cfg.experiment,"lidar_dropout") else self.cfg.experiment.lidar_dropout
        if lidar_dropout is not None:
            self.logger.info("Set LiDAR dropout to 1.0 for validation")
            self.cfg.experiment.lidar_dropout = 1.0
        
        for x_image, x_lidar, y_sequence, y_perm, image_ids in loader:
            
            batch_size = x_image.size(0) if self.cfg.experiment.encoder.use_images else x_lidar.size(0)
            
            if self.cfg.experiment.encoder.use_images:
                x_image = x_image.to(self.cfg.host.device, non_blocking=True)
            if self.cfg.experiment.encoder.use_lidar:
                x_lidar = x_lidar.to(self.cfg.host.device, non_blocking=True)    
            
            y_sequence = y_sequence.to(self.cfg.host.device, non_blocking=True)
            y_perm = y_perm.to(self.cfg.host.device, non_blocking=True)

            y_input = y_sequence[:, :-1]
            y_expected = y_sequence[:, 1:]

            preds, perm_mat = self.model(x_image, x_lidar, y_input)
            
            coords_loss = self.cfg.experiment.model.vertex_loss_weight*self.loss_fn_dict["coords"](preds.reshape(-1, preds.shape[-1]), y_expected.reshape(-1))
            
            perm_loss = self.cfg.experiment.model.perm_loss_weight*self.loss_fn_dict["perm"](perm_mat, y_perm)

            loss = coords_loss + perm_loss
            coord_loss_meter.update(coords_loss.item(), batch_size)
            perm_loss_meter.update(perm_loss.item(), batch_size)
            loss_meter.update(loss.item(), batch_size)
            
        if lidar_dropout is not None:
            self.logger.info(f"Reset LiDAR dropout to {lidar_dropout} after validation")
            self.cfg.experiment.lidar_dropout = lidar_dropout
        
        self.logger.debug(f"Validation loss: {loss_meter.global_avg:.3f}")
        loss_dict = {
            'total_loss': self.average_across_gpus(loss_meter),
            'coords_loss': self.average_across_gpus(coord_loss_meter),
            'perm_loss': self.average_across_gpus(perm_loss_meter),
        }
        self.logger.info(f"Validation loss: {loss_dict['total_loss']:.3f}")

        return loss_dict

    
    def train_one_epoch(self, epoch, iter_idx):

        # === OPTIM: empty_cache une seule fois par époque (étape 1) ===
        # Original : pas d'empty_cache du tout. On libère la mémoire CUDA fragmentée
        # au début de l'époque (pas à chaque batch, ce qui serait très coûteux).
        torch.cuda.empty_cache()

        self.logger.info(f"Train epoch {epoch}...")

        self.model.train()
        self.loss_fn_dict["coords"].train()
        self.loss_fn_dict["perm"].train()

        loss_meter = AverageMeter()
        coords_loss_meter = AverageMeter()
        perm_loss_meter = AverageMeter()
        
        loader = self.progress_bar(self.train_loader)

        for x_image, x_lidar, y_sequence, y_perm, tile_ids in loader:
            
            # self.check_y_perm(y_perm)
            # continue
            
            batch_size = x_image.size(0) if self.cfg.experiment.encoder.use_images else x_lidar.size(0)     
            
            if self.cfg.experiment.encoder.use_images:
                x_image = x_image.to(self.cfg.host.device, non_blocking=True)
            if self.cfg.experiment.encoder.use_lidar:
                x_lidar = x_lidar.to(self.cfg.host.device, non_blocking=True)
            
            y_sequence = y_sequence.to(self.cfg.host.device, non_blocking=True)
            y_perm = y_perm.to(self.cfg.host.device, non_blocking=True)

            y_input = y_sequence[:, :-1] # we do not need the last token as input, because there is no next token to predict
            y_expected = y_sequence[:, 1:] # we do not need the first token as expected, because it is always the BOS token and not predicted

            # === OPTIM: autocast float16 (étape 2) ===
            # Entoure le forward + calcul de loss en mixed precision fp16.
            # cache_enabled=False évite les fuites mémoire liées au cache d'autocast
            # avec les modules récursifs (TransformerDecoder).
            # BCELoss : perm_pred est casté en float32 explicitement car BCELoss
            # peut produire des NaN en fp16 sur certaines versions de PyTorch.
            with torch.amp.autocast('cuda', dtype=torch.float16, cache_enabled=False):
                sequence_pred, perm_pred = self.model(x_image, x_lidar, y_input)

                coords_loss = self.cfg.experiment.model.vertex_loss_weight * self.loss_fn_dict["coords"](
                    sequence_pred.reshape(-1, sequence_pred.shape[-1]), y_expected.reshape(-1)
                )
                perm_loss = self.cfg.experiment.model.perm_loss_weight * self.loss_fn_dict["perm"](
                    perm_pred.float(), y_perm.float()
                )
                loss = coords_loss + perm_loss

            # === OPTIM: zero_grad(set_to_none=True) — déjà présent, conservé (étape 1) ===
            # set_to_none=True libère la mémoire des gradients au lieu de les mettre à 0.
            self.optimizer.zero_grad(set_to_none=True)

            # === OPTIM: backward + step via GradScaler (étape 2) ===
            # scaler.scale(loss).backward() : applique un facteur d'échelle au gradient
            # pour éviter le underflow fp16. unscale_ avant step pour que les LR restent
            # cohérents. scaler.update() adapte le facteur pour les prochains batchs.
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.lr_scheduler.step()

            # === OPTIM: loss.detach() au lieu de .item() dans la boucle (étape 1) ===
            # .item() force une synchronisation CPU/GPU à chaque batch (coûteux).
            # On accumule avec .detach() et on ne fait .item() qu'à la fin via AverageMeter.
            loss_meter.update(loss.detach().item(), batch_size)
            coords_loss_meter.update(coords_loss.detach().item(), batch_size)
            perm_loss_meter.update(perm_loss.detach().item(), batch_size)

            lr = get_lr(self.optimizer)

            loader.set_postfix(train_loss=loss_meter.global_avg, lr=f"{lr:.5f}")

            iter_idx += 1
                        
        
        self.logger.debug(f"Train loss: {loss_meter.global_avg:.3f}")
        loss_dict = {
            'total_loss': self.average_across_gpus(loss_meter),
            'coords_loss': self.average_across_gpus(coords_loss_meter),
            'perm_loss': self.average_across_gpus(perm_loss_meter),
        }
        
        self.logger.info(f"Train loss: {loss_dict['total_loss']:.3f}")

        return loss_dict, iter_idx



    def train_val_loop(self):
            
        if self.cfg.checkpoint is not None:
            self.load_checkpoint()

        # === OPTIM: need_weights=False sur MultiheadAttention (étape 1) ===
        # PyTorch ne peut utiliser flash_sdp / math_sdp que si need_weights=False.
        # Par défaut les TransformerDecoderLayer ont need_weights=True ; on les force à False
        # pour tous les modules MHA du modèle (encoder ViT + decoder transformer).
        for _m in self.model.modules():
            if isinstance(_m, torch.nn.MultiheadAttention):
                _m.need_weights = False

        # === OPTIM: GradScaler AMP (étape 2) ===
        # Créé une seule fois ici, utilisé dans train_one_epoch via self.scaler.
        # Le scaler adapte dynamiquement l'échelle du gradient pour éviter le underflow fp16.
        self.scaler = torch.amp.GradScaler('cuda')

        if self.cfg.run_type.log_to_wandb and self.local_rank == 0:
            self.setup_wandb()


        # ---------- CSV LOGGER INITIALIZATION ----------
        if self.local_rank == 0:
            csv_path = os.path.join(self.cfg.output_dir, "metrics.csv")
            # Check if file exists to decide whether to write header
            file_exists = os.path.isfile(csv_path)
            self.csv_file = open(csv_path, 'a', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            if not file_exists:
                self.csv_writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_iou'])
        # ------------------------------------------------


        # === OPTIM: torch.compile (étape 3) ===
        # Appliqué après le chargement éventuel du checkpoint (les poids sont déjà en place).
        # mode="default" : bon compromis entre temps de compilation et gain de vitesse.
        # La première époque sera plus lente (compilation Triton des kernels) ;
        # le gain apparaît à partir de l'époque 2.
        # save_checkpoint utilise getattr(model, '_orig_mod', model) pour extraire les poids
        # sans le wrapper OptimizedModule (voir trainer.py).
        self.model = torch.compile(self.model, mode="default")

        iter_idx=self.cfg.experiment.model.start_epoch * len(self.train_loader)
        epoch_iterator = range(self.cfg.experiment.model.start_epoch, self.cfg.experiment.model.num_epochs)

        predictor = Predictor(self.cfg,local_rank=self.local_rank,world_size=self.world_size)

        if self.local_rank == 0:
            evaluator = Evaluator(self.cfg)
            evaluator.load_gt(self.cfg.experiment.dataset.annotations["val"])
        else:
            evaluator = None
        
        for epoch in self.progress_bar(epoch_iterator, start=self.cfg.experiment.model.start_epoch):
            
            ############################################
            ################# Training #################
            ############################################
            # important to shuffle the data differently for each epoch
            # see: https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
            if self.is_ddp:
                self.train_loader.sampler.set_epoch(epoch) 

            train_loss_dict, iter_idx = self.train_one_epoch(epoch,iter_idx)
            # Sync all processes before validation
            if self.is_ddp:
                dist.barrier()
            
            with torch.no_grad():
                if self.local_rank == 0:
                    self.visualization(self.train_loader,epoch,predictor=predictor)
                    wandb_dict ={}
                    wandb_dict['epoch'] = epoch
                    for k, v in train_loss_dict.items():
                        wandb_dict[f"train_{k}"] = v
                    wandb_dict['lr'] = get_lr(self.optimizer)


                ############################################
                ################ Validation ################
                ############################################
                val_loss_dict = self.val_one_epoch()
                if self.local_rank == 0:
                    for k, v in val_loss_dict.items():
                        wandb_dict[f"val_{k}"] = v

                #############################################
                ############## COCO Evaluation ##############
                #############################################
                val_metrics_dict = {}
                if (epoch + 1) % self.cfg.training.val_every == 0:

                    self.logger.info("Predict validation set with latest model...")
                    coco_predictions = predictor.predict_from_loader(self.model,self.tokenizer,self.val_loader)
                    
                    
                    self.logger.debug(f"rank {self.local_rank}, device: {self.device}, coco_pred_type: {type(coco_predictions)}, coco_pred_len: {len(coco_predictions)}")
                    
                    if self.is_ddp:
                        
                        # Gather the list of dictionaries from all ranks
                        gathered_predictions = [None] * self.world_size  # Placeholder for gathered objects
                        dist.all_gather_object(gathered_predictions, coco_predictions)

                        # Flatten the list of lists into a single list
                        coco_predictions = [item for sublist in gathered_predictions for item in sublist]
                    
                    if not len(coco_predictions):
                        self.logger.info("No polygons predicted. Skipping coco evaluation...")
                    else:
                        self.visualization(self.val_loader,epoch,predictor=predictor,coco_anns=coco_predictions)
                    
                    if self.local_rank == 0 and len(coco_predictions):
                        self.logger.info(f"Predicted {len(coco_predictions)}/{len(self.val_loader.dataset.coco.getAnnIds())} polygons...") 
                        self.logger.info(f"Run coco evaluation on rank {self.local_rank}...")

                        wandb_dict[f"val_num_polygons"] = len(coco_predictions)

                        prediction_outfile = os.path.join(self.cfg.output_dir, f"predictions_{self.cfg.experiment.dataset.country}_{self.cfg.evaluation.split}", f"epoch_{epoch}.json")
                        os.makedirs(os.path.dirname(prediction_outfile), exist_ok=True)
                        with open(prediction_outfile, "w") as fp:
                            fp.write(json.dumps(coco_predictions))
                        self.logger.info(f"Saved predictions to {prediction_outfile}")                        

                        evaluator.load_predictions(prediction_outfile)
                        val_metrics_dict = evaluator.evaluate()
                        evaluator.print_dict_results(val_metrics_dict)
                        
                        if val_metrics_dict['IoU'] > self.cfg.training.best_val_iou:
                            best_prediction_outfile = os.path.join(self.cfg.output_dir, f"predictions_{self.cfg.experiment.dataset.country}_{self.cfg.evaluation.split}", "best_val_iou.json")
                            shutil.copyfile(prediction_outfile, best_prediction_outfile)
                            self.logger.info(f"Copied predictions to {best_prediction_outfile}")

                        for metric, value in val_metrics_dict.items():
                            wandb_dict[f"val_{metric}"] = value
                
                
                if self.local_rank == 0:
                    self.save_best_and_latest_checkpoint(epoch, val_loss_dict, val_metrics_dict)
                    for k,v in wandb_dict.items():
                        self.logger.debug(f"{k}: {v}")
                        if self.cfg.run_type.log_to_wandb:
                            wandb.log(wandb_dict)


                    # ---------- WRITE TO CSV ----------
                    self.csv_writer.writerow([
                        epoch,
                        train_loss_dict['total_loss'],
                        val_loss_dict['total_loss'],
                        val_metrics_dict.get('IoU', float('nan'))
                    ])
                    self.csv_file.flush()
                    # ----------------------------------

                            
                # Sync all processes before next epoch
                if self.is_ddp:
                    dist.barrier()

        # after the epoch loop ends
        if self.local_rank == 0:
            self.csv_file.close()



    def check_y_perm(self, y_perm):
        
        """
        Check if y_perm is a valid permutation matrix.
        y_perm should be a binary matrix of shape [batch_size, num_vertices, num_vertices]
        """

        # Convert to int for exact checks
        xi = y_perm.to(torch.int32)

        # Row & col sum checks
        row_sums = xi.sum(dim=-1)  # shape: (16, 192)
        col_sums = xi.sum(dim=-2)  # shape: (16, 192)

        rows_per_matrix = (row_sums == 1).all(dim=1)
        cols_per_matrix = (col_sums == 1).all(dim=1)

        # Final validity
        valid_per_matrix = rows_per_matrix & cols_per_matrix
        
        assert valid_per_matrix.all(), f"Invalid permutation matrix detected! Rows per matrix: {rows_per_matrix}, Cols per matrix: {cols_per_matrix}"

    
        

