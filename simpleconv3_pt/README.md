# PyTorch Baseline
One of my favourite framework, it's logical like this:  
Zero gradient -> Forward Propagation -> Back Propagation -> Update parameters.  
And the code will be written like this:
```python
optimizer.zero_grad()
outputs = model(inputs)
loss.backward()
optimizer.step()
```
That's why I like to use it so much.  

## Contents
1. net.py: Defining network structure.
2. train.py: Training script.
3. train_with_pruning.py: Training script with pruning model.
