#!/bin/bash
#Slow models
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_3CCCM_8Units_HigherRes_Nr4.keras" --learningRate 0.001 --epochs 3
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_3CCCM_6Units_HigherRes_Nr1.keras" --learningRate 0.001 --epochs 3
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_3CCCM_6Units_HigherRes_Nr2.keras" --learningRate 0.001 --epochs 3
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_3CCCM_4Units_HigherRes_Nr4.keras" --learningRate 0.001 --epochs 3
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_3CCCM_8Units_HighRes_Nr2.keras" --learningRate 0.001 --epochs 3
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_3CCCM_8Units_HighRes_Nr3.keras" --learningRate 0.001 --epochs 5
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_3CCCM_6Units_HighRes_Nr5.keras" --learningRate 0.001 --epochs 6
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_3CCCM_6Units_HighRes_Nr6.keras" --learningRate 0.001 --epochs 6
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_4CCM_1CM_9Units_Nr3.keras" --learningRate 0.001 --epochs 3
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_4CCM_1CM_9Units_Nr4.keras" --learningRate 0.001 --epochs 6
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_4CCM_1CM_8Units_Nr1.keras" --learningRate 0.001 --epochs 6
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_4CCM_1CM_8Units_Nr4.keras" --learningRate 0.001 --epochs 6
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_4CCM_1CM_6Units_Nr1.keras" --learningRate 0.001 --epochs 6
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_4CCM_1CM_6Units_Nr2.keras" --learningRate 0.001 --epochs 9
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_4CCM_1CM_6Units_Nr3.keras" --learningRate 0.001 --epochs 9
#Tests with 2 or 0 Dense Layers - Never worked
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_4CCM_1CM_10Units_0D_Nrx.keras" --learningRate 0.001 --epochs 3
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_4CCM_1CM_6Units_2D_Nrx.keras" --learningRate 0.001 --epochs 3
# Faster models
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_5CCM_DR1_1.keras" --learningRate 0.001 --epochs 6
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_5CCM_DR1_3.keras" --learningRate 0.001 --epochs 3
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_5CCM_DR1_11.keras" --learningRate 0.001 --epochs 5
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_5CCM_DR1_12.keras" --learningRate 0.001 --epochs 5
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_5CCM_DR1_13.keras" --learningRate 0.001 --epochs 5
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_5CM_1.keras" --learningRate 0.001 --epochs 8
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_5CM_2.keras" --learningRate 0.001 --epochs 8
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_4CCM_4.keras" --learningRate 0.001 --epochs 8
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_4CCM_6.keras" --learningRate 0.001 --epochs 8
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_4CCM_8.keras" --learningRate 0.001 --epochs 8
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_4CCM_10.keras" --learningRate 0.001 --epochs 8
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_4CCM_12.keras" --learningRate 0.001 --epochs 8
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_4CCM_1CM.keras" --learningRate 0.001 --epochs 3
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_4CCM_1CM_LPQ_1.keras" --learningRate 0.001 --epochs 3
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_4CCM_1CM_LPQ_2.keras" --learningRate 0.001 --epochs 3
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_CM_4CCM_1.keras" --learningRate 0.001 --epochs 2 #4
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_CM_4CCM_2.keras" --learningRate 0.001 --epochs 2 #4
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_CM_4CCM_3.keras" --learningRate 0.001 --epochs 2 #4
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_CM_4CCM_4.keras" --learningRate 0.001 --epochs 2 #4
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_CM_4CCM_5.keras" --learningRate 0.001 --epochs 2 #4
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_CM_4CCM_6.keras" --learningRate 0.001 --epochs 2 #4
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_4CCM_1.keras" --learningRate 0.001 --epochs 8 #5
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_4CCM_2.keras" --learningRate 0.001 --epochs 8 #5
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_4CCM_13.keras" --learningRate 0.001 --epochs 8 #5
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_4CCM_14.keras" --learningRate 0.001 --epochs 8 #5
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_3CCM_2D_Nr1.keras" --learningRate 0.001 --epochs 8 #5
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_3CCM_2D_Nr2.keras" --learningRate 0.001 --epochs 8 #5
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_CM_CCM_CCCM_5.keras" --learningRate 0.001 --epochs 8 #2,5
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_CM_3CCM_1.keras" --learningRate 0.001 --epochs 8 #2,5
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_CM_3CCM_6666666.keras" --learningRate 0.001 --epochs 8 #3
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_4CCM_CM_MorePooling_Nr1.keras" --learningRate 0.001 --epochs 15
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_4CCM_CM_MorePooling_Nr2.keras" --learningRate 0.001 --epochs 3
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_4CCM_CM_MorePooling_Nr3.keras" --learningRate 0.001 --epochs 15
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_4CCM_CM_MorePooling_Nr4.keras" --learningRate 0.001 --epochs 15
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_4CCM_CM_MorePooling_Nr5.keras" --learningRate 0.001 --epochs 11
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_4CCM_CM_MorePoolingAndFilters_Nr3.keras" --learningRate 0.001 --epochs 4 # continue for a bit
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_4CCM_CM_PoolingTo11_MoreMedFilters_Nr1.keras" --learningRate 0.001 --epochs 4
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_4CCM_CM_PoolingTo11_MoreMedFilters_HR_Nr1.keras" --learningRate 0.001 --epochs 13
#python3 loadAndTrainModel.py --modelPath "model_10x240x135_4CCM_CM_MoreMedFilters_Nr1.keras" --learningRate 0.001 --epochs 4
python3 loadAndTrainModel.py --modelPath "model_10x240x135_4CCM_CM_FHHMFL_Filters_Nr1.keras" --learningRate 0.001 --epochs 4