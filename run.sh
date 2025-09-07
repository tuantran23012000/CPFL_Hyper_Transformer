cd CPFL_MOP/Connected_PF
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch
python main.py --problem ZDT2 --solver LS --mode train --model_type trans --visualize
python main.py --problem ZDT2 --solver LS --mode test --model_type trans --visualize

python main.py --problem ZDT2 --solver Cheby --mode train --model_type trans --visualize
python main.py --problem ZDT2 --solver Cheby --mode test --model_type trans --visualize