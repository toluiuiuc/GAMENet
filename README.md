# Reproducibility Project for CS598 DL4H in Spring 2023 - GAMENet: Graph Augmented MEmory Networks for Recommending Medication Combination

## About The Project

Drug recommendation is an important part of healthcare. It's important to recommend the best drugs to treat an illness while minimizing the adverse reactions one can get from a combination of medications. There are many deep learning methods that attempt to recommend a proper set of medications for a patient, however, while many of them account for temporal information about the patient, none bake in an understanding of drug-to-drug interaction (DDI) into the model.
[GAMENet](https://arxiv.org/pdf/1809.01852.pdf) remedies this by constructing a memory-based GNN model that accounts for past visits of patients using EHR, as well as a knowledge graph that represents DDI and their adverse side effects. This DDI knowledge graph is implemented as a graph convolutional network and serves as a memory model. 
This addition of a knowledge graph to model DDI, allows the model to outperform other effective Deep Learning methods in multi-label drug prediction while reducing the rate of adverse DDI interaction.

## Getting Started
1. Clone the repo
   ```sh
   git clone https://github.com/toluiuiuc/GAMENet.git
   ```
2. The project files are under **GAMENet** folder

### Requirements
1. Python >= 3.5
2. Install the required python packages by running the command below
   ```sh
   pip install -r requirements.txt
   ```

## Running the code
### Data preprocessing
In ./data, you can find the well-preprocessed data in pickle form. Also, it's easy to re-generate the data as follows:
1.  download [MIMIC data](https://mimic.physionet.org/gettingstarted/dbsetup/) and put DIAGNOSES_ICD.csv, PRESCRIPTIONS.csv, PROCEDURES_ICD.csv in ./data/
2.  download [DDI data](https://www.dropbox.com/s/8os4pd2zmp2jemd/drug-DDI.csv?dl=0) and put it in ./data/
3.  run code **./data/EDA.ipynb**

Data information in ./data:
  - records_final.pkl is the input data with four dimension (patient_idx, visit_idx, medical modal, medical id) where medical model equals 3 made of diagnosis, procedure and drug.
  - voc_final.pkl is the vocabulary list to transform medical word to corresponding idx.
  - ddi_A_final.pkl and ehr_adj_final.pkl are drug-drug adjacency matrix constructed from EHR and DDI dataset.
  - drug-atc.csv, ndc2atc_level4.csv, ndc2rxnorm_mapping.txt are mapping files for drug code transformation.

### Model Comparation
 Traning codes can be found in ./code/baseline/
 In this reproducible project, we are going to test:
 - **Nearest** will simply recommend the same combination medications at previous visit for current visit.
 - **RETAIN** can provide sequential prediction of medication combination based on a two-level neural attention model that detects influential past visits and significant clinical variables within those visits.
 

 ### GAMENet
 Going to "code" directory and run the command below:
 - Training Example
 ```
 python train_GAMENet.py --model_name GAMENet # training without DDI knowledge
 python train_GAMENet.py --model_name GAMENet --remove_dm zero_input # training without DDI knowledge and "Zero Input" method
 python train_GAMENet.py --model_name GAMENet --remove_dm remove_input # training without DDI knowledge and "Remove Input" method
 python train_GAMENet.py --model_name GAMENet --ddi # training with DDI knowledge
 python train_GAMENet.py --model_name GAMENet --ddi --remove_dm zero_input # training with DDI knowledge and "Zero Input" method
 python train_GAMENet.py --model_name GAMENet --ddi --remove_dm remove_input # training with DDI knowledge and "Remove Input" method
 ```
 - Evaluation Example
```
# General
# Testing with DDI knowledge
python train_GAMENet.py --model_name GAMENet --ddi --resume_path saved/GAMENet/Epoch_{}_JA_{}_DDI_{}.model --eval 
# Testing with DDI knowledge and "Zero Input" method
python train_GAMENet.py --model_name GAMENet --ddi --remove_dm zero_input --resume_path saved/GAMENet/Epoch_{}_JA_{}_DDI_{}.model --eval 

# Below are the pretrained examples
# Testing without DDI knowledge
python train_GAMENet.py --model_name GAMENet --resume_path pretrained/noddi.model --eval
# Testing without DDI knowledge and "Zero Input" method
python train_GAMENet.py --model_name GAMENet --resume_path pretrained/noddi_zero_input.model --remove_dm zero_input --eval
# Testing without DDI knowledge and "Remove Input" method
python train_GAMENet.py --model_name GAMENet --resume_path pretrained/noddi_remove_input.model --remove_dm remove_input --eval
# Testing with DDI knowledge
python train_GAMENet.py --model_name GAMENet --resume_path pretrained/ddi.model --ddi --eval
# Testing with DDI knowledge and "Zero Input" method
python train_GAMENet.py --model_name GAMENet --resume_path pretrained/ddi_zero_input.model --ddi --remove_dm zero_input --eval
# Testing with DDI knowledge and "Remove Input" method
python train_GAMENet.py --model_name GAMENet --resume_path pretrained/ddi_remove_input.model --ddi --remove_dm remove_input --eval
```

## Contact

TIK ON LUI - tlui2@illinois.edu
Umar Nawed - unawed2@illinois.edu

Project Link: [https://github.com/toluiuiuc/GAMENet](https://github.com/toluiuiuc/GAMENet)

## Cite 

Cite the paper:
```
@article{shang2018gamenet,
  title="{GAMENet: Graph Augmented MEmory Networks for Recommending Medication Combination}",
  author={Shang, Junyuan and Xiao, Cao and Ma, Tengfei and Li, Hongyan and Sun, Jimeng},
  journal={arXiv preprint arXiv:1809.01852},
  year={2018}
}
```