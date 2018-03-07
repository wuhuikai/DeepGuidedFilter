python prepare_dataset.py
python prepare_test_data.py
python -m hdrnet.bin.train
python -m hdrnet.bin.run
cd ../SU
source activate pytorch
python test_hdrnet.py