python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_6c233111_bn_mp143143_zp011_6c233111_bn_zp011_6c233111_bn_mp121121_zp011_6c233111_bn_zp011_6c233111_bn_mp123123_zp011_6c233111_bn_zp011_6c233111_bn_mp133133_fl_den24_do30" --learningRate 0.001 --epochs 25

python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_8c233111_bn_mp143143_zp011_8c233111_bn_zp011_8c233111_bn_mp121121_zp011_8c233111_bn_zp011_8c233111_bn_mp123123_zp011_8c233111_bn_zp011_8c233111_bn_mp133133_fl_den16_do20" --learningRate 0.001 --epochs 15

python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_10c233111_bn_mp143143_zp011_10c233111_bn_zp011_10c233111_bn_mp121121_zp011_10c233111_bn_zp011_10c233111_bn_mp123123_zp011_10c233111_bn_zp011_10c233111_bn_mp133133_fl_den8_do10" --learningRate 0.001 --epochs 25


<# 3xCCM 10=>4 #>
<#
python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_4c233111_bn_zp011_4c233111_bn_mp143143_zp011_4c233111_bn_zp011_4c233111_bn_mp143143_zp011_4c233111_bn_zp011_4c233111_bn_mp133133_fl_den32_do40" --learningRate 0.001 --epochs 15
python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_4c233111_bn_zp011_4c233111_bn_mp143143_zp011_4c233111_bn_zp011_4c233111_bn_mp143143_zp011_4c233111_bn_zp011_4c233111_bn_mp155155_fl_den96_do50" --learningRate 0.001 --epochs 15
#>


<# 3xCCM 10=>1 #>
<#
python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_4c233111_bn_zp011_4c333111_bn_mp143143_zp011_4c233111_bn_zp011_4c333111_bn_mp143143_zp011_4c233111_bn_zp011_4c333111_bn_mp133133_fl_den128_do50" --learningRate 0.001 --epochs 8
python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_4c333111_bn_zp011_4c233111_bn_mp143143_zp011_4c333111_bn_zp011_4c233111_bn_mp143143_zp011_4c333111_bn_zp011_4c233111_bn_mp133133_fl_den128_do50" --learningRate 0.001 --epochs 8
#>


<# 4xCCM 10=>2 #>
<#
python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_6c233111_bn_zp011_6c233111_bn_mp143143_zp011_6c233111_bn_zp011_6c233111_bn_mp121121_zp011_6c233111_bn_zp011_6c233111_bn_mp123123_zp011_6c233111_bn_zp011_6c233111_bn_mp133133_fl_den48_do40" --learningRate 0.001 --epochs 15
python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_4c233111_bn_zp011_4c233111_bn_mp143143_zp011_4c233111_bn_zp011_4c233111_bn_mp121121_zp011_4c233111_bn_zp011_4c233111_bn_mp123123_zp011_4c233111_bn_zp011_4c233111_bn_mp133133_fl_den64_do50" --learningRate 0.001 --epochs 15
#>


<# 3xCCCM 10=>1 #>
<#
python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_6c233111_bn_zp011_6c233111_bn_zp011_6c233111_bn_mp143143_8c233111_bn_8c233111_bn_8c233111_bn_mp133133_10c232111_bn_10c232111_bn_10c232111_bn_mp122122_fl_den48_do40" --learningRate 0.001 --epochs 9
python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_4c233111_bn_zp011_4c233111_bn_zp011_4c233111_bn_mp143143_6c233111_bn_6c233111_bn_6c233111_bn_mp133133_8c232111_bn_8c232111_bn_8c232111_bn_mp122122_fl_den64_do50" --learningRate 0.001 --epochs 9
#>


<# CM, CCM, CCCM 10=>4 #>
<#
python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_8c233111_bn_mp143143_zp011_8c233111_bn_zp011_8c233111_bn_mp143143_8c233111_bn_8c233111_bn_8c233111_bn_mp133133_fl_den32_do40" --learningRate 0.001 --epochs 15
python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_6c233111_bn_mp143143_zp011_6c233111_bn_zp011_6c233111_bn_mp143143_6c233111_bn_6c233111_bn_6c233111_bn_mp133133_fl_den48_do50" --learningRate 0.001 --epochs 15
python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_4c233111_bn_mp143143_zp011_4c233111_bn_zp011_4c233111_bn_mp143143_4c233111_bn_4c233111_bn_4c233111_bn_mp133133_fl_den96_do50" --learningRate 0.001 --epochs 15
python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_4c233111_bn_mp143143_zp011_6c233111_bn_zp011_6c233111_bn_mp143143_8c233111_bn_8c233111_bn_8c233111_bn_mp133133_fl_den48_do50" --learningRate 0.001 --epochs 15
python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_4c233111_bn_mp143143_zp011_8c233111_bn_zp011_8c233111_bn_mp143143_12c233111_bn_12c233111_bn_12c233111_bn_mp133133_fl_den24_do30" --learningRate 0.001 --epochs 15
#>


<# CM, CCCM, CCCM 10=>3 #>
<#
python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_8c233111_bn_mp143143_zp011_8c233111_bn_zp011_8c233111_bn_zp011_8c233111_bn_mp143143_8c233111_bn_8c233111_bn_8c233111_bn_mp133133_fl_den48_do50" --learningRate 0.001 --epochs 25
python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_4c233111_bn_mp143143_zp011_6c233111_bn_zp011_6c233111_bn_zp011_6c233111_bn_mp143143_8c233111_bn_8c233111_bn_8c233111_bn_mp133133_fl_den48_do50" --learningRate 0.001 --epochs 15
python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_4c233111_bn_mp143143_6c233111_bn_6c233111_bn_6c233111_bn_mp133133_8c232111_bn_8c232111_bn_8c232111_bn_mp122122_fl_den16_do20" --learningRate 0.001 --epochs 15
python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_4c233111_bn_mp143143_8c233111_bn_8c233111_bn_8c233111_bn_mp133133_12c232111_bn_12c232111_bn_12c232111_bn_mp122122_fl_den8_do10" --learningRate 0.001 --epochs 15
python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_4c233111_bn_mp143143_6c244111_bn_6c244111_bn_6c244111_bn_mp133133_8c244111_bn_8c244111_bn_8c244111_bn_mp143143_fl_den128_do50" --learningRate 0.001 --epochs 15
#>


<# 2xCCM, 1xCCCM 10=>3 #>
<#
python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_12c233111_bn_zp011_12c233111_bn_mp143143_zp011_12c233111_bn_zp011_12c233111_bn_mp143143_12c233111_bn_12c233111_bn_12c233111_bn_mp133133_fl_den16_do20" --learningRate 0.001 --epochs 4

python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_12c233111_bn_zp011_12c233111_bn_mp143143_zp011_12c233111_bn_zp011_12c233111_bn_mp143143_12c233111_bn_12c233111_bn_12c233111_bn_mp133133_fl_den8_do10" --learningRate 0.001 --epochs 4
#>

<# CM, CCM, CCCM 10=>4? #>
<#
python loadAndTrainModel.py --modelPath "model_10x240x135_zp011_12c233111_bn_mp143143_zp011_12c233111_bn_zp011_12c233111_bn_mp143143_12c233111_bn_12c233111_bn_12c233111_bn_mp133133_fl_den8_do10" --learningRate 0.001 --epochs 2
#>


<# --evaluate #>
<#
python loadAndTrainModel.py --modelPath "cp_0_1482_model_10x240x135_zp011_12c233111_bn_zp011_12c233111_bn_mp143143_zp011_12c233111_bn_zp011_12c233111_bn_mp143143_zp011_12c233111_bn_zp011_12c233111_bn_mp155155_fl_den16_do20" --learningRate 0.001 --epochs 1 --evaluate
#>


<#
sora ai videos

Sensitivity-Slider
val 0 => img 366 9783 9733

testen, ob trainingsexmaples exakt gespeichert werden (bytes statt ints)
#>

<#

(1) 0.1875: cM   CCM  CCM    => 3 3 16 => 8(10)
(2) 0.1960: cCM  CCM  CCM    => 3 3 16 => 8(10)
(3) 0.1962: CcCM CCCM        => 3 3 16 => 8(10)
(3) 0.2025: cCM  CCM  CCM    => 3 3 16 => 8(10)		255
(4) 0.2098: CcM  CCM  CCM    => 3 3 16 => 8(10)
(5) 0.2287: cCM  CCM         => 3 3 16 => 8(10)
(6) 0.2482: cM   CCM  CCM    => 3 3 16 => 4(0)
(7) 0.2493: CM   CCM  CCM CM => 1 1 36 => 16(20)

3xCCM
1xCM, 2xCCM
1xCM, 2xCCM, 1xCM
3xCM
2xCCM, 1xCC
2xCCCM
4xCC bzw. 3xCC, 1xC NoMP

3xCCM 255
1xCM, 2xCCM 255
2xCCCM 255

3xCCM 10=>4
3xCCM 10=>1
4xCCM 10=>2
3xCCCM 10=>1
2xCCM, 1xCCCM 10=>3
CM, CCM, CCCM 10=>3

#>