# Experiments-to-run.
# Comment those done

# Permutations: Architecture, Regularization, Activation Function, Optimizer
1. Lenet, Dropout, Tanh, Adam                   # Base
2. Lenet, Dropout = 0, Tanh, Adam               # No dropout
3. Lenet, Dropout, Tanh, RMSProp                # RMSProp Optimizer
4. Lenet, Dropout, ReLU, Adam                   # ReLU activation
5. Lenet, Batch Normalization, Tanh, Adam       # Batch normalization 
6. Inception, Dropout, <std activation>, Adam   # Base
7. Inception, Dropout = 0, <std activation>, Adam  # No Dropout
8. Inception, Dropout, <std activation>, RMSProp   # RMSProp optimizer
9. Inception, Dropout, <std activation>, SGD    # SGD optimizer
10. Inception, Batch Normalization, <std activation>, Adam  # Batch normalization (IF POSSIBLE)