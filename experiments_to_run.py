# Experiments-to-run.
# Comment those done

# Permutations: Architecture, Regularization, Activation Function, Optimizer
# ~ 1. Lenet, No Dropout, Tanh, SGD                 # Base
2. Lenet, Dropout, Tanh, SGD                    # Dropout
3. Lenet, Batch Normalization, Tanh, SGD        # Batch normalization
# ~ 4. Lenet, No Dropout, Tanh, RMSProp             # RMSProp Optimizer
5. Lenet, No Dropout, ReLU, SGD                 # ReLU activation
# ~ 6. Lenet, No Dropout, Tanh, Adam                # Adam Optimizer
7. Inception, Dropout, std, Adam                # Base
8. Inception, Dropout = 0, std, Adam            # No Dropout
9. Inception, Batch Normalization, std, Adam    # Batch normalization (IF POSSIBLE)
10. Inception, Dropout, std, RMSProp            # RMSProp optimizer
11. Inception, Dropout, std, SGD                # SGD optimizer


## Results
1 loss: 0.7672, acc: 0.7435
4 loss: 2.5020, acc: 0.7620
6 loss: 1.5865, acc: 0.7778
