from src.forProp import ForwardFeed_Test
from argparse import ArgumentParser
import pickle as p
import random as r


def main():
    parser = ArgumentParser(description='Testing the weights and bias values generated from train.py')
    parser.add_argument('test_data', type=str, help='weights and bias file created from training')
    parser.add_argument('weights_file', type=str, help='weights and bias file created from training')
    parser.add_argument('--index', type=int, default=r.randint(1,100), help='select random value from test dataset')   
    args = parser.parse_args()


    with open(args.weights_file, 'rb') as f:
        wb = p.load(f)
    

    fft = ForwardFeed_Test(args.test_data, wb, args.index)
    A1 = fft.HL1_fw() #Input layer
    A2 = fft.HL2_fw(A1) #First hidden layer
    A3 = fft.HL3_fw(A2) #Second hidden layer


    label, pred = fft.predict(A3)
    print(label)
    print(pred)  
    fft.imShow()

if __name__ == '__main__':
    main()