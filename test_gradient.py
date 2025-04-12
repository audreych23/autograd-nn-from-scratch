import numpy as np
import HW2 as hw2

def numerical_gradient(f, inputs, eps=1e-5):
    """
    Numerically estimate gradients using central differences.
    `f` should be a function that takes a list of numpy arrays and returns a scalar.
    `inputs` is a list of Variable objects.
    """
    grads = []
    for var in inputs:
        grad = np.zeros_like(var.data)
        it = np.nditer(var.data, flags=['multi_index'], op_flags=['readwrite'])

        while not it.finished:
            idx = it.multi_index

            orig = var.data[idx]
            
            var.data[idx] = orig + eps
            f_pos = f([v.data for v in inputs])
            
            var.data[idx] = orig - eps
            f_neg = f([v.data for v in inputs])

            grad[idx] = (f_pos - f_neg) / (2 * eps)
            var.data[idx] = orig
            it.iternext()
        
        grads.append(grad)
    return grads


def test_gradient_check():
    np.random.seed(42)
    model = hw2.MLP(2, 2)
    x_np = np.array([[0.1, 0.2], [0.3, 0.5]])
    y_np = hw2.one_hot(np.array([[1], [0]]), 2)
    print(y_np)
    x = hw2.Variable(x_np)
    y_true = hw2.Variable(y_np)

    def loss_fn(numpy_params):
        # Update model parameters with given numpy values
        for p, val in zip(model.parameters(), numpy_params):
            p.data = val.copy()

        # Forward pass
        y_pred = model(x)
        criterion = hw2.CategoricalCrossEntropyLoss()
        loss = criterion(y_pred, y_true)
        return loss.data

    # Forward & backward with autograd
    y_pred = model(x)
    criterion = hw2.CategoricalCrossEntropyLoss()
    loss = criterion(y_pred, y_true)
    loss.backward()

    # Get analytical gradients
    autograd_grads = [p.grad.copy() for p in model.parameters()]

    # Compare with numerical gradients
    numeric_grads = numerical_gradient(loss_fn, model.parameters())

    for i, (a, n) in enumerate(zip(autograd_grads, numeric_grads)):
        print(f"\nParam {i}")
        print("Analytical grad:\n", a)
        print("Numerical grad:\n", n)
        diff = np.abs(a - n)
        print("Diff:\n", diff)
        print("Max diff:", np.max(diff))

if __name__ == "__main__":
    test_gradient_check()