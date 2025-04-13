import numpy as np
from PIL import Image
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# https://www.youtube.com/watch?v=dB-u77Y5a6A&t=1604s - Reference
# =============== SEED ====================
np.random.seed(42)

# ======================================= Builds Computation Graph ================================================================
class Variable:
    """
    Helps in building a computational graph
    """
    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self._backward = lambda: None # backward is guaranteed to be not none after an operation / forward pass is called  
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Variable(data={self.data}, grad={self.grad})"
    
    def backward(self, gradient=None):
        if gradient is None:
            if np.prod(self.data.shape) == 1:
                gradient = np.ones_like(self.data)
            else:
                raise RuntimeError("Gradient must be specified for non scalar output, for scalar output, it is automatically set to be 1")
    
        # Accumulate the gradient but when do I get the gradient what
        self.grad += gradient

        # topoological ordering from loss fn to input (back to front)
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # node here is the function that could correspond to sum, multiply
        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        """
        Reset the gradient to zero
        """
        self.grad = np.zeros_like(self.data)

    def __add__(self, other):
        # forward pass
        other = other if isinstance(other, Variable) else Variable(other)
        upstream = Variable(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += upstream.grad
            # Local gradient for addition is 1 for both inputs

            other.grad += unbroadcast(upstream.grad, other.data.shape)

        upstream._backward = _backward
        return upstream
    
    def __sub__(self, other):
        return self + (-other)
    
    def __neg__(self):
        return self * -1
    
    def __mul__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        upstream = Variable(self.data * other.data, (self, other), '*')

        def _backward():
            # local gradient is the other value (the opposite va;ue) 
            local_grad_self = other.data
            local_grad_other = self.data
            grad_self =  local_grad_self * upstream.grad
            grad_other = local_grad_other * upstream.grad

            self.grad += unbroadcast(grad_self, self.data.shape)
            other.grad += unbroadcast(grad_other, other.data.shape)
            # self.grad += grad_self
            # other.grad += grad_other

        upstream._backward = _backward
        return upstream
    
    def __pow__(self, power):
        assert isinstance(power, (int, float)), "Power must be a scalar"
        out = Variable(self.data ** power, (self,), f'**{power}')
        
        def _backward():
            # Local gradient: power * x^(power-1)
            self.grad += (power * self.data ** (power - 1)) * out.grad
        
        out._backward = _backward
        return out
    
    def __matmul__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        upstream = Variable(self.data @ other.data, (self, other), '@')

        def _backward():
            # The gradients for matrix multiplication needs a little trick  
            # [https://www.youtube.com/watch?v=dB-u77Y5a6A&t=1604s]
            # x : [N x D] w : [D x M] y : [N x M]
            self.grad += upstream.grad @ other.data.T # N X D = [N X M] [M x D]
            other.grad += self.data.T @ upstream.grad # D x M = [D x N] [N x M]

        upstream._backward = _backward
        return upstream
    
    def exp(self):
        upstream = Variable(np.exp(self.data), (self,), 'exp')
        
        def _backward():
            # Local gradient: exp(x)
            self.grad += upstream.data * upstream.grad
        
        upstream._backward = _backward
        return upstream

    def sum(self, dim=None):
        upstream = Variable(np.sum(self.data, axis=dim), (self,), 'sum')
        
        def _backward():
            # Create gradient with proper shape for broadcasting
            grad = np.ones_like(self.data) * upstream.grad
            if dim is not None:
                grad = np.expand_dims(grad, axis=dim)
            self.grad += grad
        
        upstream._backward = _backward
        return upstream
    
    def log(self):
        upstream = Variable(np.log(self.data), (self,), 'log')
        
        def _backward():
            # Local gradient: 1/x
            self.grad += (1.0 / self.data) * upstream.grad
        
        upstream._backward = _backward
        return upstream
    
    def clip(self, min_val, max_val):
        clipped_data = np.clip(self.data, min_val, max_val)
        upstream = Variable(clipped_data, (self,), 'clip')

        def _backward():
            mask = (self.data >= min_val) & (self.data <= max_val)
            self.grad += mask * upstream.grad

        upstream._backward = _backward
        return upstream

# ======================================= Layer Class (ReLu, SoftMax, Linear/ Dense layer) ================================================================
class Layer():
    def __call__(self, x):
        return self.forward(x)
    
    # forward has to be implemented
    def forward(self, x):
        raise NotImplementedError
    
    def parameters(self):
        return []

class ReLU(Layer):
    def forward(self, x):
        upstream = Variable(np.maximum(0, x.data), (x,), 'ReLU')

        def _backward():
            x.grad += (x.data > 0) * upstream.grad

        upstream._backward = _backward
        return upstream
    
    def parameters(self):
        return []


def softmax(x, dim=-1):
    """
    Softmax activation function (numerically stable implementation)
    Computes softmax along the specified dimension.
    """
    # Shift input for numerical stability
    shifted_x = x.data - np.max(x.data, axis=dim, keepdims=True)
    exp_x = np.exp(shifted_x)
    softmax_output = exp_x / np.sum(exp_x, axis=dim, keepdims=True)
    
    upstream = Variable(softmax_output, (x,), 'softmax')
    # print(out)
    
    # If we combine softmax + CCE loss the backward is very simple and it becomes (probabilities - targets) / batch_size
    def _backward():
        # Trust in the process that softmax will also only get gradient from CCE 
        # if not then it probably breaks TODO: USe other methods 
        x.grad += upstream.grad
    
    upstream._backward = _backward
    return upstream

class Softmax(Layer):
    def __init__(self, dim=1):
        self.dim = dim

    def forward(self, x):
        return softmax(x, dim=self.dim)
    
    def parameters(self):
        return []


class Linear(Layer):
    def __init__(self, in_features, out_features):
        # Xavier initialization 
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.weight = Variable(np.random.randn(in_features, out_features) * scale)
        self.bias = Variable(np.zeros(out_features))

    def forward(self, x):
        # no need for backward since we're using in built autograph function
        return x @ self.weight + self.bias 
    
    def parameters(self):
        return [self.weight, self.bias]



# ======================================= Loss Function ================================================================

class MSELoss:
    def __call__(self, y_pred, y_true):
        return ((y_pred - y_true) ** 2).sum()
    
class CategoricalCrossEntropyLoss:
    def __init__(self, apply_softmax=True, dim=-1, eps=1e-12):
        self.dim = dim
        self.eps = eps  # Small epsilon to avoid log(0)
    
    def __call__(self, probs, targets):
        batch_size = probs.data.shape[0]
        
        # Clip probabilities to avoid numerical issues
        probs_clipped = np.clip(probs.data, self.eps, 1.0 - self.eps)
        
        # Check if targets are class indices or one-hot encoded
        if len(targets.data.shape) == 1 or targets.data.shape[1] == 1:
            # Targets are class indices, convert to one-hot
            num_classes = probs.data.shape[1]
            one_hot = np.zeros((batch_size, num_classes))
            for i, idx in enumerate(targets.data.flatten().astype(int)):
                one_hot[i, idx] = 1
            targets_one_hot = one_hot
        else:
            # Targets are already one-hot encoded
            targets_one_hot = targets.data
        
        # Compute cross entropy loss
        log_probs = np.log(probs_clipped)
        loss = -np.sum(targets_one_hot * log_probs) / batch_size
        
        # Create output variable
        out = Variable(loss, (probs,), 'cross_entropy')
        
        def _backward():
            # Gradient of cross entropy with respect to softmax output
            # is (probabilities - targets) / batch_size
            gradient = (probs.data - targets_one_hot) / batch_size
            probs.grad += unbroadcast(gradient, probs.data.shape)

        out._backward = _backward
        return out
    

# ======================================= Optimizer SGD ================================================================
class SGD:
    def __init__(self, parameters, lr=0.01, momentum=0.0, weight_decay=0.0):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [np.zeros_like(param.data) for param in parameters]
    
    def step(self):
        for i, param in enumerate(self.parameters):
            # Apply weight decay (L2 regularization)
            if self.weight_decay > 0:
                param.grad += self.weight_decay * param.data
            
            # Update velocity with momentum
            self.velocity[i] = self.momentum * self.velocity[i] - self.lr * param.grad
            
            # Update parameters
            param.data += self.velocity[i]
    
    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()


# ======================================= Model Defintion: 2-layer MLP ================================================================
class MLP:
    # input size is the amount of inputs features, output size is the amount of class output
    def __init__(self, input_size, output_size):
        self.linear1 = Linear(input_size, 10)
        self.relu = ReLU()
        self.linear2 = Linear(10, output_size)
        self.softmax = Softmax()


    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

    def parameters(self):
        return self.linear1.parameters() + self.linear2.parameters()

# ======================================= Utility Function ================================================================
def one_hot(labels, num_classes):
    labels = np.asarray(labels).reshape(-1).astype(int)  # (N,)
    return np.eye(num_classes)[labels]        # (N, C)

def unbroadcast(grad, target_shape):
    """Sum grad to match the shape of target (reverse broadcasting)."""
    while len(grad.shape) > len(target_shape):
        grad = grad.sum(axis=0)

    for i, (gdim, tdim) in enumerate(zip(grad.shape, target_shape)):
        if gdim != tdim:
            grad = grad.sum(axis=i, keepdims=True)

    return grad

# test if the dimensions make sense
def test_autograph():
    # Dummy input and labels
    x = Variable([[1.0, 2.0], [1.5, 3.0], [2.0, 3.0]])  # shape: [2, 2]
    y_true_np = Variable(one_hot(np.array([[1], [1], [0]]), 2))  # shape: [2, 1]
    # y_true_np = Variable(np.array([[1], [1], [0]]))


    # Model & loss
    model = MLP(2, 2)
    criterion = CategoricalCrossEntropyLoss()
    # criterion = MSELoss()

    # Forward pass
    y_pred = model(x)
    loss = criterion(y_pred, y_true_np)

    print("Loss:", loss.data)

    # Backward pass
    loss.backward()

    # Print gradients
    for i, param in enumerate(model.parameters()):
        print(f"Grad {i} shape: {param.grad.shape}\n{param.grad}\n {param._op}")

    criterion.zero_grad()

def run_test():
    # Dummy input and labels
    x = Variable([[1.0, 2.0], [1.5, 3.0], [1.0, 6.0]])  # shape: [2, 2]
    y_true_np = Variable(one_hot(np.array([[1], [1], [0]]), 2))  # shape: [2, 1]
    # y_true_np = Variable(np.array([[1], [1], [0]]))

    x_test = Variable([[1.0, 2.0], [1.5, 3.0]])
    y_test = Variable(np.array([[1], [1]]))
    model = MLP(2, 2)
    criterion = CategoricalCrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr = 0.01)

    EPOCHS = 300

    # Train model
    for epoch in range(EPOCHS):
        # for batch_X, batch_y in loader:
        optimizer.zero_grad()
        y_pred = model(x)
        print(y_pred)
        loss = criterion(y_pred, y_true_np)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.data}")

    # Evaluate the model 
    outputs = model(x_test)
    print(model.parameters())
    print("out", outputs)
    predicted = np.argmax(outputs.data, axis=1)
    print(predicted.shape)
    corrected = 0
    print(predicted)
    print(y_test.data.reshape(-1))
    corrected = (predicted == y_test.data.reshape(-1)).sum()
    print(corrected)


# =============================================== Data Loader ================================================
class DataLoader:
    def __init__(self, x, y, batch_size=32, shuffle=True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(x))
        self.reset()

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_idx = 0

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        if self.current_idx >= len(self.x):
            raise StopIteration

        idx = self.indices[self.current_idx: self.current_idx + self.batch_size]
        x_batch = self.x[idx]
        y_batch = self.y[idx]
        self.current_idx += self.batch_size
        return Variable(x_batch), Variable(y_batch)

def load_images_from_dir_labels(dir, size=(32, 32)):
    X = []
    y = []
    class_names = sorted(os.listdir(dir))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for class_name in class_names:
        class_folder = os.path.join(dir, class_name)
        if not os.path.isdir(class_folder):
            continue
        for fname in os.listdir(class_folder):
            if fname.endswith(('.png')):
                path = os.path.join(class_folder, fname)
                # it's actually already 32 x 32, but just in case, we're gonna resize it again here, just in case if test data 
                img = Image.open(path).resize(size)
                img_array = np.array(img, dtype=np.float32).flatten() / 255.0
                X.append(img_array)
                y.append(class_to_idx[class_name])

    return np.stack(X), np.array(y), class_to_idx

def evaluate_accuracy(test_data_loader):
    num_correct = 0
    num_total = 0

    for x_batch_test, y_batch_test in test_data_loader:
        y_pred = model(x_batch_test)
        y_pred = np.argmax(y_pred.data, axis=1)
        correct = (y_pred == y_batch_test.data.reshape(-1)).sum()
        num_correct += correct
        num_total += y_batch_test.data.shape[0]
    
    accuracy = num_correct / num_total
    return accuracy

if __name__ == "__main__":        
    # ======== param ===========
    BATCH_SIZE = 32
    EPOCHS = 100
    NUM_CLASSES = 3

    # ======== Setup train and test data ===========================
    # train data 
    X_train, y_train, dict_class_idx = load_images_from_dir_labels("Data_train")
    # do pca here for X_train (REQUIRED) (The data has been scaled to range 0, 1, but before using pca we can make it 0 mean and unit variance)
    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train)
    pca = PCA(n_components = 2)
    X_pca_train = pca.fit_transform(X_scaled_train)
    y_train = y_train.reshape(-1, 1)

    # test data
    X_test, y_test, dict_class_idx = load_images_from_dir_labels("Data_test")
    # do pca here for X_test (REQUIRED)
    X_scaled_test = scaler.transform(X_test)
    X_pca_test = pca.transform(X_scaled_test)
    y_test = y_test.reshape(-1, 1)

    train_data_loader = DataLoader(X_pca_train, y_train)
    test_data_loader = DataLoader(X_pca_test, y_test, batch_size=BATCH_SIZE, shuffle=False)

    # ======== model, optimizer, criterion ============================
    # There will be 2 features after doing PCA     
    model = MLP(2, 3)
    criterion = CategoricalCrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr = 0.01)
    # ==================== train ======================================
    for epoch in range(EPOCHS):
        for x_batch, y_batch in train_data_loader:
            optimizer.zero_grad()
            y_train_one_hot = one_hot(y_batch.data, NUM_CLASSES)
            
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_train_one_hot)

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.data}")

    # =================== evaluate ====================================
    test_accuracy = evaluate_accuracy(test_data_loader)
    print("test_accuracy:", test_accuracy)
