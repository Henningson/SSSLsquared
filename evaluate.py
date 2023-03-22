import torch
import albumentations as A
import os
import Utils.utils as utils
from Metrics.Evaluator import * # Fix this
from Metrics.Timing import * # Fix this
from typing import List
from torch.utils.data import DataLoader
from models.LSQ import LSQLocalization

import sys
sys.path.append("models/")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Input list of every model path that should be checked inside a class
# For example [[UNET_A, UNET_B, UNET_C], [OURS_A, OURS_B, OURS_C], [..]]
# Outputs [[[Precision, F1-Score, NME, IOU, DICE, InferenceSpeed], [..]] for every supplied model]
def evaluate_everything(checkpoints: List[List[str]], dataset_path: str, group_names: List[str]) -> List[List[List[float]]]:
    group_evals = []
    group_times = []
    for MODEL_GROUP in checkpoints:
        per_model_evals = []
        per_model_times = []
        for CHECKPOINT_PATH in MODEL_GROUP:
            if CHECKPOINT_PATH == "" or not os.path.isdir(CHECKPOINT_PATH):
                print("\033[93m" + "Please provide a viable checkpoint path")

            config = utils.load_config(os.path.join(CHECKPOINT_PATH, "config.yml"))
            config["dataset_path"] = dataset_path
            val_transforms = A.load(os.path.join(CHECKPOINT_PATH, "val_transform.yaml"), data_format='yaml')
            neuralNet = __import__(config["model"])
            model = neuralNet.Model(config, state_dict=torch.load(os.path.join(CHECKPOINT_PATH, "model.pth.tar"))).to(DEVICE)
            dataset = __import__('dataset').__dict__[config['dataset_name']]

    
            try:
                old_sequence_length = config["sequence_length"]
                config["sequence_length"] = 1
            except:
                old_sequence_length = 1
                config["sequence_length"] = 1


            val_ds = dataset(config, is_train=False, transform=val_transforms)
            val_loader = DataLoader(val_ds, 
                                    batch_size=config["batch_size"]*old_sequence_length,
                                    num_workers=2, 
                                    pin_memory=True, 
                                    shuffle=False)
            
            localizer = LSQLocalization(local_maxima_window = 7, 
                                        gauss_window = config["gauss_window"], 
                                        heatmapaxis = config["heatmapaxis"], 
                                        threshold = 0.7,
                                        device=DEVICE)
            
            config["sequence_length"] = old_sequence_length
            evaluator = None
            metrics = [PrecisionMetric(), F1ScoreMetric(), NMEMetric(), DiceMetric(), JaccardIndexMetric()]

            if config["model"] == "TwoDtoThreeDNet" or config["model"] == "TwoDtoThreeDNet2":
                timer_eval = InferenceTimer2D3D(model, val_loader, localizer, config)
                evaluator = Evaluator2D3D(model, val_loader, localizer, config, metrics)
            else:
                timer_eval = BaseInferenceTimer(model, val_loader, localizer, config)
                evaluator = BaseEvaluator(model, val_loader, localizer, config, metrics)


            with torch.no_grad():
                print("#"*20)
                print("#"*3 + " " + CHECKPOINT_PATH + " " + "#"*3)
                print("#"*20)
                timer_eval.evaluate()
                evaluator.evaluate()
                
                per_model_times.append(timer_eval.get_total_time())
                per_model_evals.append(evaluator.get_final_metrics())
        group_evals.append(per_model_evals)
        group_times.append(per_model_times)

    for group_name, group_scores, group_time in zip(group_names, group_evals, group_times):
        print("############" + group_name + "############")
        print("Precision, F1-Score, NME, DICE, IoU")
        print(torch.tensor(group_scores).mean(dim=0))
        print(torch.tensor(group_scores).std(dim=0))
        print("Inference-Time")
        print(torch.tensor(group_time).mean())
        print(torch.tensor(group_time).std())



def evaluate_single_network(checkpoint_path):
    # TODO: Implement me
    pass


def main():
    
    UNET_FULL = ["checkpoints/UNETFULL_CFCM_2558", 
                    "checkpoints/UNETFULL_DDFH_1010", 
                    "checkpoints/UNETFULL_LSRH_1355", 
                    "checkpoints/UNETFULL_MKMS_9400", 
                    "checkpoints/UNETFULL_SSTM_5445"]

    LSTM_UNET = ["checkpoints/LSTM_MK_MS_7289",
                "checkpoints/LSTM_CF_CM_5010", 
                "checkpoints/LSTM_DD_FH_7886", 
                "checkpoints/LSTM_LS_RH_8013", 
                "checkpoints/LSTM_SS_TM_6150"]

    ZweiDDreiD = ["checkpoints/2D3D_CFCM_01_9160", 
                "checkpoints/2D3D_DDFH_01_3749", 
                "checkpoints/2D3D_LSRH_01_6746", 
                "checkpoints/2D3D_MKMS_01_3279", 
                "checkpoints/2D3D_SSTM_01_650"]
    
    MODEL_GROUPS = [UNET_FULL, ZweiDDreiD, LSTM_UNET]
    MODEL_GROUP_NAMES = ["UNET_FULL", "ZweiDDreiD", "LSTM_UNET"]

    evaluate_everything(MODEL_GROUPS, '../HLEDataset/dataset/', MODEL_GROUP_NAMES)

if __name__ == "__main__":
    main()