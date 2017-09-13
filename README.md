# OpenNMT: Open-Source Neural Machine Translation

This is my personalized dev. prototype for NMT research. Great thanks to [Pytorch](https://github.com/pytorch/pytorch) port of [OpenNMT](https://github.com/OpenNMT/OpenNMT), an open-source (MIT) neural machine translation system.

<center style="padding: 40px"><img width="70%" src="http://opennmt.github.io/simple-attn.png" /></center>

## Quickstart

## Some useful tools:

The example below uses the Moses tokenizer (http://www.statmt.org/moses/) to prepare the data and the moses BLEU script for evaluation.

```bash
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/tokenizer.perl
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.de
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.en
sed -i "s/$RealBin\/..\/share\/nonbreaking_prefixes//" tokenizer.perl
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl
```

## IWSLT 2014 German-English

### 0) Dataset. 

The original dataset has been cleaned, and split into `train` and `dev` data; the `test` data is processed in the same way.  

### 1) Preprocess the data.

```bash
python preprocess.py -train_src IWSLT/train.de.tok -train_tgt IWSLT/train.en.tok -valid_src IWSLT/dev.de.tok -valid_tgt IWSLT/dev.en.tok -save_data IWSLT/de2en.30k
```

### 2) Train the model.

```bash
# de2en
python train.py -data IWSLT/de2en.30k.train.pt -save_model ../Models/V1_IWSLT_Models/de2en_30k_bz32_bleuEval -gpus 0 -batch_size 32

# with bleu eval
python train.py -data IWSLT/de2en.30k.train.pt -save_model ../Models/V1_IWSLT_Models/de2en_30k_bz64_bc3  -devSrcPath IWSLT/dev.de.tok -devTgtPath IWSLT/dev.en.tok -gpus 0 -batch_size 64

## debug
python train.py -data IWSLT/de2en.30k.train.pt -save_model ../Models/V1_IWSLT_Models/de2en_30k_bz32_bleuEval  -devSrcPath IWSLT/test.de.small.tok -devTgtPath IWSLT/dev.en.tok -gpus 0 -batch_size 32
```

### 3) Translate sentences.

```bash
python translate.py -gpu 3 -model ../Models/V1_IWSLT_Models/de2en_30k_bz32_bc2_bleu_25.93_e13.pt -src IWSLT/test.de.tok -tgt IWSLT/test.en.tok -replace_unk -verbose -output iwslt_pred/de2en_bc2_pred.txt
```

### 4) Evaluate.

```bash
perl multi-bleu.perl IWSLT/test.en.tok < iwslt_pred/de2en_30k_bz32_pred.txt
```

## LDC Chinese-English

### 0) Dataset

[TO-DO] Say something

### 1) Preprocess

```bash
python preprocess.py -train_src ../Datasets/LDC/train.cn.tok -train_tgt ../Datasets/LDC/train.en.tok -valid_src IWSLT/dev.de.tok -valid_tgt IWSLT/dev.en.tok -save_data ../Datasets/LDC/cn2en.30k
```