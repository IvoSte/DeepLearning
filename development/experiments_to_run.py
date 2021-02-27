# Experiments-to-run.
# Comment those done

# Permutations: Architecture, Regularization, Activation Function, Optimizer
# ~ 1. Lenet, No Dropout, Tanh, SGD                 # Base
# ~ 2. Lenet, Dropout, Tanh, SGD                    # Dropout layer between F5 and F6
# ~ 3. Lenet, Batch Normalization, Tanh, SGD        # Batch normalization
# ~ 4. Lenet, No Dropout, Tanh, RMSProp             # RMSProp Optimizer
# ~ 5. Lenet, No Dropout, ReLU, SGD                 # ReLU activation
# ~ 6. Lenet, No Dropout, Tanh, Adam                # Adam Optimizer
# ~ 7. Lenet, Double Dropout, Tanh, SGD				# Second dropout layer between F6 and F7
# ~ 8. Lenet, No Dropout, Tanh, Adadelta			# Adadelta Optimizer

# nope
7. Inception, Dropout, std, Adam                # Base
8. Inception, Dropout = 0, std, Adam            # No Dropout
9. Inception, Batch Normalization, std, Adam    # Batch normalization (IF POSSIBLE)
10. Inception, Dropout, std, RMSProp            # RMSProp optimizer
11. Inception, Dropout, std, SGD                # SGD optimizer


## Results from evaluation
1 loss: 0.7315, acc: 0.7676
2 loss: 0.8122, acc: 0.6361
3 loss: 0.9370, acc: 0.6991
4 loss: 2.5063, acc: 0.7574
5 loss: 0.7264, acc: 0.7833
6 loss: 1.2753, acc: 0.7806
7 loss: 0.7917, acc: 0.6426
8 loss: 0.9668, acc: 0.5231
