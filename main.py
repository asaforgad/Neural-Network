import scipy.io
import numpy as np
import copy

from part_1 import gradient_test, test_sgd_least_squares, sgd, find_best_learning_rates_and_batch_sizes
from neural_network import NeuralNetwork
from residual_network import ResNet
from tests import (
    gradient_test_network,
    jacobian_test_network
)


def loadDataset(Data):
    try:
        if Data == 'GMMData':
            data = scipy.io.loadmat('GMMData.mat')
            print("GMM Data loaded.")
        elif Data == 'PeaksData':
            data = scipy.io.loadmat('PeaksData.mat')
            print("Peaks Data loaded.")
        elif Data == 'SwissRollData':
            data = scipy.io.loadmat('SwissRollData.mat')
            print("Swiss Roll Data loaded.")
        else:
            print("Unknown dataset selected.")
            return None, None, None, None

        x_train = data['Yt']
        y_train = data['Ct']
        x_val = data['Yv']
        y_val = data['Cv']
        return x_train, y_train, x_val, y_val
    except FileNotFoundError:
        print(f"File for {Data} not found.")
        return None, None, None, None
    except KeyError as e:
        print(f"Missing key in the dataset: {e}")
        return None, None, None, None


def get_valid_input(prompt, num_options):
    while True:
        try:
            choice = int(input(prompt))
            if 1 <= choice <= num_options:
                return choice
            else:
                print(f"Invalid choice. Please select a number between 1 and {num_options}.")
        except ValueError:
            print("Enter a number.")


if __name__ == "__main__":
    while True:
        tasks = ['Part 1', 'Part 2', 'Exit']
        print("\nChoose task to test:")
        for i, task in enumerate(tasks):
            print(f"{i + 1}) {task}")
        
        task_choice = get_valid_input("-> ", len(tasks))
        
        if task_choice == 3:
            print("Exiting\n")
            break
        
        if task_choice == 1:
            sections = [
                'Gradient Test (Softmax Regression)', 
                'Test Stochastic Gradient Descent, Least Squares', 
                'Test Stochastic Gradient Descent, Softmax', 
                'Return to Main Menu'
            ]
            print("\nChoose section:")
            for i, section in enumerate(sections):
                print(f"{i + 1}) {section}")
            section_choice = get_valid_input("-> ", len(sections))
            
            if section_choice == 4:
                continue  
            
            if section_choice == 1:
                gradient_test()
            elif section_choice == 2:
                test_sgd_least_squares()
            elif section_choice == 3:
                data_options = ['GMMData', 'PeaksData', 'SwissRollData']
                print("\nChoose a dataset:")
                for i, option in enumerate(data_options):
                    print(f"{i + 1}) {option}")
                
                data_choice = get_valid_input("-> ", len(data_options))
                chosen_data = data_options[data_choice - 1]
                x_train, y_train, x_val, y_val = loadDataset(chosen_data)
                
                if x_train is None:
                    continue  
                
                params_options = [
                    'Default', 
                    'Find best learning rate and batch size for SGD', 
                    'Customize',
                    'Return to Previous Menu'
                ]
                print("\nChoose a parameter option:")
                for i, option in enumerate(params_options):
                    print(f"{i + 1}) {option}")
                
                params_choice = get_valid_input("-> ", len(params_options))
                
                if params_choice == 4:
                    continue  
                
                if params_choice == 1:
                    sgd(
                        x_train.T, 
                        y_train.T, 
                        learning_rate=0.01, 
                        batch_size=64, 
                        epochs=500, 
                        x_val=x_val.T, 
                        y_val=y_val.T,
                        plot=True,
                        print_progress=True
                    )
                elif params_choice == 2:
                    print("Finding best learning rate and batch size for SGD for 50 iterations.")
                    find_best_learning_rates_and_batch_sizes(
                        x_train.T, 
                        y_train.T, 
                        x_val.T, 
                        y_val.T, 
                        iterations=50, 
                        plot_graph=True, 
                        print_each_iteration=True
                    )
                elif params_choice == 3:
                    while True:
                        try:
                            learning_rate = float(input("Enter the learning rate: -> "))
                            break
                        except ValueError:
                            print("Invalid input. Please enter a number for the learning rate.")
                    
                    while True:
                        try:
                            batch_size = int(input("Enter the batch size: -> "))
                            break
                        except ValueError:
                            print("Invalid input. Please enter a number for the batch size.")
                    
                    sgd(
                        x_train.T, 
                        y_train.T, 
                        learning_rate=learning_rate, 
                        batch_size=batch_size, 
                        epochs=500, 
                        x_val=x_val.T, 
                        y_val=y_val.T,
                        plot=True,
                        print_progress=True
                    )
            else:
                print("Invalid choice. Please select a valid option.")
                continue 

        elif task_choice == 2:
            data_options = ['GMMData', 'PeaksData', 'SwissRollData']
            print("\nChoose a dataset:")
            for i, option in enumerate(data_options):
                print(f"{i + 1}) {option}")
            
            data_choice = get_valid_input("-> ", len(data_options))
            chosen_data = data_options[data_choice - 1]
            x_train, y_train, x_val, y_val = loadDataset(chosen_data)
            
            if x_train is None:
                continue
            
            print("\nChoose a network type:")
            network_options = ["Neural Network", "Residual Network"]
            for i, option in enumerate(network_options):
                print(f"{i + 1}) {option}")
            
            network_choice = get_valid_input("-> ", len(network_options))
            chosen_network = network_options[network_choice - 1]
            
            print("\nChoose a network structure:")
            structure_options = ["Default", "Customize", "Return to Main Menu"]
            for i, option in enumerate(structure_options):
                print(f"{i + 1}) {option}")
            
            structure_choice = get_valid_input("-> ", len(structure_options))
            if structure_choice == 3:
                continue  
            
            if structure_choice == 1:
                num_epochs = 500
                if chosen_network == "Neural Network":
                    if chosen_data == "GMMData":
                        layer_sizes = [x_train.shape[0], 22, y_train.shape[0]]
                        model = NeuralNetwork(layer_sizes, lr=0.2)
                        batch_size = 200
                    elif chosen_data == "PeaksData":
                        layer_sizes = [x_train.shape[0], 14, 14, y_train.shape[0]]
                        model = NeuralNetwork(layer_sizes, lr=0.05)
                        batch_size = 200
                    elif chosen_data == "SwissRollData":
                        layer_sizes = [x_train.shape[0], 14, 14, y_train.shape[0]]
                        model = NeuralNetwork(layer_sizes, lr=0.05)
                        batch_size = 200
                elif chosen_network == "Residual Network":
                    if chosen_data == "GMMData":
                        input_dim = x_train.shape[0]
                        output_dim = y_train.shape[0]
                        model = ResNet(input_dim, [20, 20], output_dim, lr=0.2)
                        batch_size = 200
                    elif chosen_data == "PeaksData":
                        input_dim = x_train.shape[0]
                        output_dim = y_train.shape[0]
                        model = ResNet(input_dim, [14], output_dim, lr=0.07)
                        batch_size = 200
                    elif chosen_data == "SwissRollData":
                        input_dim = x_train.shape[0]
                        output_dim = y_train.shape[0]
                        model = ResNet(input_dim, [20, 20], output_dim, lr=0.03)
                        batch_size = 200
                print(f'\nThe network has {model.num_layers} layers with activation functions tanh for hidden layers and softmax for output layer and learning rate {model.lr}.') 
            elif structure_choice == 2:
                while True:
                    try:
                        lr = float(input("Enter the learning rate: -> "))
                        break
                    except ValueError:
                        print("Invalid input. Please enter a number for the learning rate.")
                
                while True:
                    try:
                        num_hidden_layers = int(input("Enter the number of hidden layers: -> "))
                        if num_hidden_layers < 0:
                            print("Number of hidden layers cannot be negative.")
                            continue
                        break
                    except ValueError:
                        print("Invalid input. Please enter a number for the number of hidden layers.")
                
                hidden_layer_sizes = []
                activations = []
                print(f"\nThe first layer input size is {x_train.shape[0]}")
                for i in range(num_hidden_layers):
                    while True:
                        try:
                            size = int(input(f"Enter the output size of hidden layer {i + 1}: -> "))
                            hidden_layer_sizes.append(size)
                            break
                        except ValueError:
                            print("Invalid input. Please enter a number for the hidden layer size.")
                    
                    while True:
                        activation = input(f"Enter the activation function for hidden layer {i + 1} (1 for tanh, 2 for relu): -> ")
                        if activation == "1":
                            activations.append("tanh")
                            break
                        elif activation == "2":
                            activations.append("relu")
                            break
                        else:
                            print("Invalid choice. Please enter 1 for tanh or 2 for relu.")
                
                activations.append("softmax")
                if chosen_network == "Neural Network":
                    layer_sizes = [x_train.shape[0]] + hidden_layer_sizes + [y_train.shape[0]]
                    model = NeuralNetwork(layer_sizes, activations=activations, lr=lr)
                    batch_size = 200
                elif chosen_network == "Residual Network":
                    input_dim = x_train.shape[0]
                    output_dim = y_train.shape[0]
                    model = ResNet(input_dim, hidden_layer_sizes, output_dim, activations=activations, lr=lr)
                    batch_size = 200
                print(f'\nThe last layer input size is {x_train.shape[0]}, the hidden layer sizes are {hidden_layer_sizes}, the output size is {y_train.shape[0]}, the activation functions are {activations} and the learning rate is {model.lr}.')
                print("Customized network created successfully.")
            else:
                print("Invalid choice. Please select a valid option.")
                continue  
            
            while True:
                print("\nChoose an operation to perform:")
                test_options = ["Gradient Test", "Jacobian Test", "Train", "Sample 200 random data points","Return to Main Menu"]
                for i, option in enumerate(test_options):
                    print(f"{i + 1}) {option}")
                
                operation_choice = get_valid_input("-> ", len(test_options))
                
                if operation_choice == 5:
                    break  
                
                chosen_choice = test_options[operation_choice - 1]
                
                
                if chosen_choice == "Train":
                    while True:
                        try:
                            num_epochs = int(input("Enter the number of epochs - 0 for default (500): -> "))
                            if num_epochs < 0:
                                print("Number of epochs cannot be negative.")
                                continue
                            if num_epochs == 0:
                                num_epochs = 500
                            break
                        except ValueError:
                            print("Invalid input. Please enter a number for the number of epochs.")
                    
                    while True:
                        try:
                            batch_size_input = int(input("Enter the batch size - 0 for default (200): -> "))
                            if batch_size_input < 0:
                                print("Batch size cannot be negative.")
                                continue
                            if batch_size_input == 0:
                                batch_size_input = 200
                            break
                        except ValueError:
                            print("invalid input. Please enter a number for the batch size.")
                    
                    model_copy = copy.deepcopy(model) 
                    model_copy.train(x_train, y_train, num_epochs=num_epochs, batch_size=batch_size_input, x_val=x_val, y_val=y_val)
                
                elif chosen_choice == "Gradient Test":
                    if chosen_network == "Neural Network":
                        gradient_test_network(model, x_train, y_train, num_samples=10)
                    elif chosen_network == "Residual Network":
                        gradient_test_network(model, x_train, y_train, num_samples=10)
                
                elif chosen_choice == "Jacobian Test":
                    if chosen_network == "Neural Network":
                        jacobian_test_network(model, x_train, sample_num=10)
                    elif chosen_network == "Residual Network":
                        jacobian_test_network(model, x_train, sample_num=10)
                
                elif chosen_choice == "Sample 200 random data points":
                    indices = np.random.choice(x_train.shape[1], 200, replace=False)
                    x_sample = x_train[:, indices]
                    y_sample = y_train[:, indices]
                    #train the model
                    model.forward(x_sample)
                    model.train(x_sample, y_sample, num_epochs=500, batch_size=100, x_val=x_val, y_val=y_val)
                    exit()
                
                else:
                    print("Invalid operation. Please select a valid option.")
        
        else:
            print("Invalid choice. Please select a valid option.")