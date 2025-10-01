pip install tensorflow[and-cuda]=="2.15.0.post1"

pip install hydra-core=="1.3.2"

# this is needed for cub-200 dataset
pip install -U tfds-nightly
pip install opencv-python-headless=="4.9.0.80"

# this is needed for transformer architecture
pip install ml-collections=="0.1.1"
pip install tensorflow-hub=="0.14.0"
pip install huggingface-hub=="0.17.3"
pip install transformers=="4.34.0"
# in the following we use --no-deps because we do not want to make tfds updated
pip install --no-deps keras-cv=="0.8.2"
pip install keras-core=="0.1.7"

# results analysis
pip install pandas=="2.3.3"