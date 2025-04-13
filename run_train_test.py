import numpy as np
import HW2 as hw2

def run_test():
    # Dummy input and labels
    x = hw2.Variable([[1.0, 2.0], [1.5, 3.0], [1.0, 6.0]])  # shape: [2, 2]
    y_true_np = hw2.Variable(hw2.one_hot(np.array([[1], [1], [0]]), 2))  # shape: [2, 1]
    # y_true_np = Variable(np.array([[1], [1], [0]]))

    x_test = hw2.Variable([[1.0, 2.0], [1.5, 3.0]])
    y_test = hw2.Variable(np.array([[1], [1]]))
    model = hw2.MLP(2, 2)
    criterion = hw2.CategoricalCrossEntropyLoss()
    optimizer = hw2.SGD(model.parameters(), lr = 0.01)

    EPOCHS = 10

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

if __name__ == "__main__":
    run_test()