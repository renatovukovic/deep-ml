import numpy as np

class DropoutLayer:
    def __init__(self, p: float):
        """Initialize the dropout layer.
        
        Attributes to set:
            self.p: the dropout rate
            self.mask: stores the dropout mask (initially None)
        """
        # Your code here
        self.p = p
        self.mask = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass of the dropout layer.
        
        Generate a new mask on each training forward pass and store it in self.mask.
        """
        # Your code here
        if training:
            #create binary mask with 1 keep and 0 dropped to multiply with input
            self.mask = np.random.rand(*x.shape) < (1 - self.p) #* for unpacking the shape tuple

            #multiply with mask and divide by 1-p to keep the mean
            x_dropped = (x * self.mask) / (1-self.p)

            return x_dropped

        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass of the dropout layer.
        
        Use the stored self.mask from the most recent forward pass.
        """
        # Your code here
        #multply the grad with the mask to only update the parameters that were not dropped and scale with 1-p
        if self.mask is not None:
            grad_dropped = (grad * self.mask) / (1 - self.p)

            return grad_dropped
        
        else:
            return grad