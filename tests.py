from matplotlib import pyplot as plt
import numpy as np

from functions import cross_entropy_loss
from layers import Layer, ResNetLayer


# Gradient test for the whole network
def gradient_test_network(model, X, y, num_samples=1, epsilon_iterator=[0.5 ** i for i in range(1, 11)]):
    Y = np.argmax(y, axis=0)  # Convert one-hot to integer labels
    zero_order_errors = np.zeros(len(epsilon_iterator))
    first_order_errors = np.zeros(len(epsilon_iterator))

    is_resNet = False
    output = model.forward(X)   # Forward pass
    F0 = cross_entropy_loss(output, Y) # Loss without perturbation
    grads = model.backward(X, Y, output) # Backward pass (compute gradients)

    # Flatten gradients
    flat_grads = []
    for idx, layer in enumerate(model.layers, start=1):
        if layer.residental == False:
            flat_grads.append(grads.get(f'W{idx}', np.zeros_like(layer.W)).flatten())
            flat_grads.append(grads.get(f'b{idx}', np.zeros_like(layer.b)).flatten())
        else:
            is_resNet = True
            flat_grads.append(grads.get(f'W1{idx}', np.zeros_like(layer.W1)).flatten())
            flat_grads.append(grads.get(f'b1{idx}', np.zeros_like(layer.b1)).flatten())
            flat_grads.append(grads.get(f'W2{idx}', np.zeros_like(layer.W2)).flatten())
            flat_grads.append(grads.get(f'b2{idx}', np.zeros_like(layer.b2)).flatten())
    flat_grads = np.concatenate(flat_grads)

    original_params = model.get_flat_params().copy()
    
    for idx in range(num_samples):
        perturbations = np.random.randn(len(flat_grads))
        perturbations /= np.linalg.norm(perturbations) if np.linalg.norm(perturbations) != 0 else 1.0

        for k, eps in enumerate(epsilon_iterator):
            flat_params_plus = original_params + eps * perturbations
            model.set_flat_params(flat_params_plus)

            output_perturbation = model.forward(X)
            F_plus = cross_entropy_loss(output_perturbation, Y) # Loss with perturbation

            zero_order_error = np.abs(F_plus - F0)
            first_order_error = np.abs(F_plus - F0 - eps * np.dot(flat_grads, perturbations))

            zero_order_errors[k] += zero_order_error / num_samples
            first_order_errors[k] += first_order_error / num_samples

        model.set_flat_params(original_params)

    plt.figure(figsize=(8, 6))
    plt.plot(epsilon_iterator, zero_order_errors, label="Zero order error")
    plt.plot(epsilon_iterator, first_order_errors, label="First order error")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Epsilon")
    plt.ylabel("Error")
    if is_resNet:
        plt.title("Gradient test - Residual Network")
    else:
        plt.title("Gradient test - Netural Network")
    plt.legend()
    plt.gca().invert_xaxis()
    plt.show()

    return zero_order_errors, first_order_errors

def jacobian_test_network(model, X, sample_num=1, epsilon_iterator = [0.5 ** i for i in range(1, 11)], plot=True):
    model.forward(X)
    layers_num = model.num_layers
    for i in range(layers_num-1):
        curr_layer = model.layers[i]
        v = np.random.rand(curr_layer.output.shape[0], curr_layer.input.shape[1])
        v /= np.linalg.norm(v) if np.linalg.norm(v) != 0 else 1
        X = curr_layer.input.astype(np.float64)  # Avoid overflow
        
        base_forward = np.vdot(v, curr_layer.output)
                
        if curr_layer.residental: 
            gard_x, grad_w1, grad_b1, grad_w2, grad_b2 = curr_layer.backward(v)
            
            # Testing W1
            test_gradients(curr_layer.W1, grad_w1, "W1", sample_num=sample_num,epsilon_iterator=epsilon_iterator, plot=plot, layerNum=i+1, X=X, v=v, curr_layer=curr_layer, base_forward=base_forward)
            # Testing b1
            test_gradients(curr_layer.b1, grad_b1, "b1", sample_num=sample_num,epsilon_iterator=epsilon_iterator, plot=plot, layerNum=i+1, X=X, v=v, curr_layer=curr_layer, base_forward=base_forward)
            # Testing W2
            test_gradients(curr_layer.W2, grad_w2, "W2", sample_num=sample_num,epsilon_iterator=epsilon_iterator, plot=plot, layerNum=i+1, X=X, v=v, curr_layer=curr_layer, base_forward=base_forward)
            # Testing b2
            test_gradients(curr_layer.b2, grad_b2, "b2", sample_num=sample_num,epsilon_iterator=epsilon_iterator, plot=plot, layerNum=i+1, X=X, v=v, curr_layer=curr_layer, base_forward=base_forward)
            # Testing X
            test_gradients(curr_layer.input, gard_x, "X", sample_num=sample_num,epsilon_iterator=epsilon_iterator, plot=plot, layerNum=i+1, X=X, v=v, curr_layer=curr_layer, base_forward=base_forward)
        else:   
            grad_x, grad_w, grad_b = curr_layer.backward(v)
             
            # Testing W
            test_gradients(curr_layer.W, grad_w, "W", sample_num=sample_num,epsilon_iterator=epsilon_iterator, plot=plot, layerNum=i+1, X=X, v=v, curr_layer=curr_layer, base_forward=base_forward)
            # Testing b
            test_gradients(curr_layer.b, grad_b, "b", sample_num=sample_num,epsilon_iterator=epsilon_iterator, plot=plot, layerNum=i+1, X=X, v=v, curr_layer=curr_layer, base_forward=base_forward)
            # Testing X
            test_gradients(curr_layer.input, grad_x, "X", sample_num=sample_num,epsilon_iterator=epsilon_iterator, plot=plot, layerNum=i+1, X=X, v=v, curr_layer=curr_layer, base_forward=base_forward)
            
def test_gradients(parameter, grad_param, param_name, sample_num,epsilon_iterator, plot, layerNum, X, v, curr_layer, base_forward):
    # Initialize accumulators for differences
    accum_grad_diffs = np.zeros(len(epsilon_iterator))
    accum_grad_diffs_grad = np.zeros(len(epsilon_iterator))
                    
    for i in range(sample_num):
        # Generate a random perturbation
        perturbations = np.random.randn(*parameter.shape)
        perturbations /= np.linalg.norm(perturbations) if np.linalg.norm(perturbations) != 0 else 1
                        
        original_param = parameter.copy()
                        
        for idx, eps in enumerate(epsilon_iterator):
            # Perturb the parameter
            parameter += perturbations * eps
            # Forward pass after perturbation
            forward_after_eps = np.vdot(v, curr_layer.forward(X))
            # Revert the parameter to original
            parameter[:] = original_param
                            
            # Compute differences
            diff = np.abs(forward_after_eps - base_forward)
            grad_diff = np.abs(forward_after_eps - base_forward - np.vdot(grad_param, perturbations * eps))
                            
            # Accumulate differences
            accum_grad_diffs[idx] += diff
            accum_grad_diffs_grad[idx] += grad_diff
                    
    # Compute average differences
    avg_grad_diffs = accum_grad_diffs / sample_num
    avg_grad_diffs_grad = accum_grad_diffs_grad / sample_num
                    
    if plot:
        plt.plot(epsilon_iterator, avg_grad_diffs, label=f"Difference without grad ({param_name})")
        plt.plot(epsilon_iterator, avg_grad_diffs_grad, label=f"Difference with grad ({param_name})")
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel('Average Difference')
        plt.xlabel('Epsilon')
        plt.title(f'Average Difference vs. Epsilon for {param_name}, Layer: {layerNum}')
        plt.legend()
        plt.gca().invert_xaxis()
        plt.show()
                        
        return avg_grad_diffs, avg_grad_diffs_grad