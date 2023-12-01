#Install Packages
python -m pip install -r requirements.txt

#Install llama-cpp-python
python -m pip uninstall llama-cpp-python -y
CMAKE_ARGS="-DLLAMA_METAL=on" pip install -U llama-cpp-python --no-cache-dir
python -m pip install 'llama-cpp-python[server]'

