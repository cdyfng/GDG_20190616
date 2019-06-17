conda info --env
conda activate darkflow	#(My env for pic process)
pip install keras==2.1.4
pip install pydot_ng
brew install graphviz
pip install imsave
pip install scipy
pip install --upgrade pip
pip install pillow
pip3 install scikit-image

python transform.py -i ../rHuang1.jpeg -s la_muse -b 0.1 -o  out
python transform.py -i ./ZiqiandHan.jpg  -s la_muse -b 0.1 -o  out
