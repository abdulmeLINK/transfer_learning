# Test AlexNet with all scenarios
python main.py --model AlexNet --scenario 1
python main.py --model AlexNet --scenario 2
python main.py --model AlexNet --scenario 3
python main.py --model AlexNet --scenario 4

# Test ResNet with all scenarios
python main.py --model ResNet --scenario 1
python main.py --model ResNet --scenario 2
python main.py --model ResNet --scenario 3
python main.py --model ResNet --scenario 4

# Test VGG with all scenarios
python main.py --model VGG --scenario 1
python main.py --model VGG --scenario 2
python main.py --model VGG --scenario 3
python main.py --model VGG --scenario 4

# For AlexNet
python main.py --model AlexNet --scenario 1 --bypass_train
python main.py --model AlexNet --scenario 2 --bypass_train
python main.py --model AlexNet --scenario 3 --bypass_train
python main.py --model AlexNet --scenario 4 --bypass_train

# For ResNet
python main.py --model ResNet --scenario 1 --bypass_train
python main.py --model ResNet --scenario 2 --bypass_train
python main.py --model ResNet --scenario 3 --bypass_train
python main.py --model ResNet --scenario 4 --bypass_train

# For VGG
python main.py --model VGG --scenario 1 --bypass_train
python main.py --model VGG --scenario 2 --bypass_train
python main.py --model VGG --scenario 3 --bypass_train
python main.py --model VGG --scenario 4 --bypass_train