

python loadAndTrainModel.py --modelPath "model_10x240x135_5CCM_DR1" --learningRate 0.001 --epochs 10
python loadAndTrainModel.py --modelPath "model_10x240x135_5CCM_DR1_3" --learningRate 0.001 --epochs 10



<#
python loadAndTrainModel.py --modelPath "model_10x240x135_1CCCM_2CCM_2CM" --learningRate 0.001 --epochs 5
python loadAndTrainModel.py --modelPath "model_10x240x135_4CCM_1CM" --learningRate 0.001 --epochs 17
python loadAndTrainModel.py --modelPath "model_10x240x135_4CCM_4" --learningRate 0.001 --epochs 4
python loadAndTrainModel.py --modelPath "model_10x240x135_4CCM_6" --learningRate 0.001 --epochs 4
python loadAndTrainModel.py --modelPath "model_10x240x135_4CCM_8" --learningRate 0.001 --epochs 4
python loadAndTrainModel.py --modelPath "model_10x240x135_4CCM_10" --learningRate 0.001 --epochs 4
python loadAndTrainModel.py --modelPath "model_10x240x135_4CCM_12" --learningRate 0.001 --epochs 2

#>





<# Transformer-Models #>
<#
python loadAndTrainModel.py --modelPath "model_10x240x135_transf2" --learningRate 0.001 --epochs 2
#>

<# CM, 4xCCM 10=>2 #>
<#
python loadAndTrainModel.py --modelPath "model_10x240x135_CM_4CCM_DR2_1" --learningRate 0.001 --epochs 4
#>

<# 5xCM 10=>6 #>
<#
python loadAndTrainModel.py --modelPath "model_10x240x135_5CM_DR6_1" --learningRate 0.001 --epochs 4
#>

<# 5xCM 10=>5 #>
<#
python loadAndTrainModel.py --modelPath "model_10x240x135_5CM_1" --learningRate 0.001 --epochs 4
python loadAndTrainModel.py --modelPath "model_10x240x135_5CM_2" --learningRate 0.001 --epochs 4
#>

<# CM, 4xCCM 10=>1 #>
<#
python loadAndTrainModel.py --modelPath "model_10x240x135_CM_4CCM_4" --learningRate 0.001 --epochs 4
python loadAndTrainModel.py --modelPath "model_10x240x135_CM_4CCM_5" --learningRate 0.001 --epochs 4
#>



<# CM, 2xCCCM 10=>3 #>
<#
python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_4c233111_bn_mp143143_zp011_6c233111_bn_zp011_6c233111_bn_zp011_6c233111_bn_mp143143_8c233111_bn_8c233111_bn_8c233111_bn_mp133133_fl_den48_do50" --learningRate 0.001 --epochs 2

python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_8c233111_bn_mp143143_zp011_8c233111_bn_zp011_8c233111_bn_zp011_8c233111_bn_mp143143_8c233111_bn_8c233111_bn_8c233111_bn_mp133133_fl_den48_do50" --learningRate 0.001 --epochs 2

python loadAndTrainModel.py --modelPath "model_10x240x135_CM_2CCCM_3" --learningRate 0.001 --epochs 2
python loadAndTrainModel.py --modelPath "model_10x240x135_CM_2CCCM_4" --learningRate 0.001 --epochs 2
python loadAndTrainModel.py --modelPath "model_10x240x135_CM_2CCCM_5" --learningRate 0.001 --epochs 2
#>



<# CM, 3xCCM 10=>3 #>
<#
python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_6c233111_bn_mp143143_zp011_6c233111_bn_zp011_6c233111_bn_mp121121_zp011_6c233111_bn_zp011_6c233111_bn_mp123123_zp011_6c233111_bn_zp011_6c233111_bn_mp133133_fl_den24_do30" --learningRate 0.001 --epochs 5

python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_6c233111_bn_mp123123_zp011_6c233111_bn_zp011_6c233111_bn_mp121121_zp011_6c233111_bn_zp011_6c233111_bn_mp143143_zp011_6c233111_bn_zp011_6c233111_bn_mp155155_fl_den64_do50" --learningRate 0.001 --epochs 5
#>


<# CM, CCM, CCCM 10=>4 #>
<#
python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_8c233111_bn_mp143143_zp011_8c233111_bn_zp011_8c233111_bn_mp143143_8c233111_bn_8c233111_bn_8c233111_bn_mp133133_fl_den32_do40" --learningRate 0.001 --epochs 10

python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_6c233111_bn_mp143143_zp011_6c233111_bn_zp011_6c233111_bn_mp143143_6c233111_bn_6c233111_bn_6c233111_bn_mp133133_fl_den48_do50" --learningRate 0.001 --epochs 10

python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_4c233111_bn_mp143143_zp011_6c233111_bn_zp011_6c233111_bn_mp143143_8c233111_bn_8c233111_bn_8c233111_bn_mp133133_fl_den48_do50" --learningRate 0.001 --epochs 6

python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_4c233111_bn_mp143143_zp011_8c233111_bn_zp011_8c233111_bn_mp143143_12c233111_bn_12c233111_bn_12c233111_bn_mp133133_fl_den24_do30" --learningRate 0.001 --epochs 6
#>


<# 2xCCM, 1xCCCM 10=>3 #>
<#
python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_12c233111_bn_zp011_12c233111_bn_mp143143_zp011_12c233111_bn_zp011_12c233111_bn_mp143143_12c233111_bn_12c233111_bn_12c233111_bn_mp133133_fl_den16_do20" --learningRate 0.001 --epochs 2

python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_12c233111_bn_zp011_12c233111_bn_mp143143_zp011_12c233111_bn_zp011_12c233111_bn_mp143143_12c233111_bn_12c233111_bn_12c233111_bn_mp133133_fl_den8_do10" --learningRate 0.001 --epochs 2
#>


<# 3xCCCM 10=>1 #>
<#
python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_6c233111_bn_zp011_6c233111_bn_zp011_6c233111_bn_mp143143_8c233111_bn_8c233111_bn_8c233111_bn_mp133133_10c232111_bn_10c232111_bn_10c232111_bn_mp122122_fl_den48_do40" --learningRate 0.001 --epochs 9
python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_4c233111_bn_zp011_4c233111_bn_zp011_4c233111_bn_mp143143_6c233111_bn_6c233111_bn_6c233111_bn_mp133133_8c232111_bn_8c232111_bn_8c232111_bn_mp122122_fl_den64_do50" --learningRate 0.001 --epochs 9
#>


<# 4xCCM 10=>2 #>
<#
python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_4c233111_bn_zp011_4c233111_bn_mp143143_zp011_4c233111_bn_zp011_4c233111_bn_mp121121_zp011_4c233111_bn_zp011_4c233111_bn_mp123123_zp011_4c233111_bn_zp011_4c233111_bn_mp133133_fl_den64_do50" --learningRate 0.001 --epochs 2
python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_4c233111_bn_zp011_4c233111_bn_mp143143_zp011_4c233111_bn_zp011_4c233111_bn_mp121121_zp011_4c233111_bn_zp011_4c233111_bn_mp123123_zp011_4c233111_bn_zp011_4c233111_bn_mp133133_fl_den64_do50_2" --learningRate 0.001 --epochs 2
python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_4c233111_bn_zp011_4c233111_bn_mp143143_zp011_4c233111_bn_zp011_4c233111_bn_mp121121_zp011_4c233111_bn_zp011_4c233111_bn_mp123123_zp011_4c233111_bn_zp011_4c233111_bn_mp133133_fl_den64_do50_3" --learningRate 0.001 --epochs 2
python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_4c233111_bn_zp011_4c233111_bn_mp143143_zp011_4c233111_bn_zp011_4c233111_bn_mp121121_zp011_4c233111_bn_zp011_4c233111_bn_mp123123_zp011_4c233111_bn_zp011_4c233111_bn_mp133133_fl_den96_do50" --learningRate 0.001 --epochs 2
python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_4c233111_bn_zp011_4c233111_bn_mp143143_zp011_4c233111_bn_zp011_4c233111_bn_mp121121_zp011_4c233111_bn_zp011_4c233111_bn_mp123123_zp011_4c233111_bn_zp011_4c233111_bn_mp133133_fl_den48_do50" --learningRate 0.001 --epochs 2
#>


<# 3xCCM 10=>1 #>
<#
python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_4c233111_bn_zp011_4c333111_bn_mp143143_zp011_4c233111_bn_zp011_4c333111_bn_mp143143_zp011_4c233111_bn_zp011_4c333111_bn_mp133133_fl_den128_do50" --learningRate 0.001 --epochs 6
python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_4c333111_bn_zp011_4c233111_bn_mp143143_zp011_4c333111_bn_zp011_4c233111_bn_mp143143_zp011_4c333111_bn_zp011_4c233111_bn_mp133133_fl_den128_do50" --learningRate 0.001 --epochs 6
#>


<# 3xCCM 10=>4 #>
<#
python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_4c233111_bn_zp011_4c233111_bn_mp143143_zp011_4c233111_bn_zp011_4c233111_bn_mp143143_zp011_4c233111_bn_zp011_4c233111_bn_mp133133_fl_den32_do40" --learningRate 0.001 --epochs 6
python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_4c233111_bn_zp011_4c233111_bn_mp143143_zp011_4c233111_bn_zp011_4c233111_bn_mp143143_zp011_4c233111_bn_zp011_4c233111_bn_mp155155_fl_den96_do50" --learningRate 0.001 --epochs 6
#>







<# --evaluate example #>
<#
python loadAndTrainModel.py --modelPath "cp_0_1482_model_10x240x135_zp011_12c233111_bn_zp011_12c233111_bn_mp143143_zp011_12c233111_bn_zp011_12c233111_bn_mp143143_zp011_12c233111_bn_zp011_12c233111_bn_mp155155_fl_den16_do20" --learningRate 0.001 --epochs 1 --evaluate
#>
