from torch import nn

from torch.autograd import Function

class DiffX(Function):
    def __init__(self, r):
        super(DiffX, self).__init__()
        self.r = r

    def forward(self, input):
        assert input.dim() == 4

        r = self.r
        output = input.new().resize_as_(input)

        output[:,:,   :r+1] = input[:,:,    r:2*r+1]
        output[:,:,r+1: -r] = input[:,:,2*r+1:     ] - input[:,:,      :-2*r-1]
        output[:,:, -r:   ] = input[:,:,   -1:     ] - input[:,:,-2*r-1:  -r-1]

        return output

    def backward(self, grad_output):
        r = self.r
        grad_input  = grad_output.new().resize_as_(grad_output).zero_()

        grad_input[:,:,     r: 2*r+1] += grad_output[:,:,   :r+1]
        grad_input[:,:, 2*r+1:      ] += grad_output[:,:,r+1:-r ]
        grad_input[:,:,      :-2*r-1] -= grad_output[:,:,r+1:-r ]
        grad_input[:,:,    -1:      ] += grad_output[:,:, -r:   ].sum(dim=2, keepdim=True)
        grad_input[:,:,-2*r-1:  -r-1] -= grad_output[:,:, -r:   ]

        return grad_input

def diff_x(input, r):
    return DiffX(r)(input)

class DiffY(Function):
    def __init__(self, r):
        super(DiffY, self).__init__()
        self.r = r

    def forward(self, input):
        assert input.dim() == 4

        r = self.r
        output = input.new().resize_as_(input)

        output[:,:,:,   :r+1] = input[:,:,:,    r:2*r+1]
        output[:,:,:,r+1: -r] = input[:,:,:,2*r+1:     ] - input[:,:,:,      :-2*r-1]
        output[:,:,:, -r:   ] = input[:,:,:,   -1:     ] - input[:,:,:,-2*r-1:  -r-1]

        return output

    def backward(self, grad_output):
        r = self.r
        grad_input  = grad_output.new().resize_as_(grad_output).zero_()

        grad_input[:,:,:,     r: 2*r+1] += grad_output[:,:,:,   :r+1]
        grad_input[:,:,:, 2*r+1:      ] += grad_output[:,:,:,r+1:-r ]
        grad_input[:,:,:,      :-2*r-1] -= grad_output[:,:,:,r+1:-r ]
        grad_input[:,:,:,    -1:      ] += grad_output[:,:,:, -r:   ].sum(dim=3, keepdim=True)
        grad_input[:,:,:,-2*r-1:  -r-1] -= grad_output[:,:,:, -r:   ]

        return grad_input

def diff_y(input, r):
    return DiffY(r)(input)

class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()

        self.r = r

    def forward(self, x):
        assert x.dim() == 4

        return diff_y(diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)