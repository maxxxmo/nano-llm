from src.model.mlp import MLP
from src.model.optimizer import Adam
from src.model.activation import Softmax 

model = MLP(3, [4, 4, 3], dtype='float32')
optimizer = Adam(model.parameters(), lr=0.01) 
softmax = Softmax()

# Data
xs = [
    [2.0, 3.0, 1.0], 
    [-3.0, -1.0, -0.5], 
    [-0.5, -1.0, -1.0], 
    [1.0, 1.0, 1.0], 
    [-0.5, 1.0, -1.0], 
    [1.0, -1.0, 1.0]
]
ys = [
    [1, 0, 0], 
    [0, 1, 0], 
    [0, 1, 0], 
    [1, 0, 0], 
    [0, 0, 1], 
    [0, 0, 1]
]

# Training
for epoch in range(100):
    # --- Forward Pass ---
    ypred_raw = [model(x) for x in xs] # logits
    ypred_probs = [softmax(y) for y in ypred_raw] # probs
    # --- Loss ---
    loss = sum(
        sum((yout_i - ygt_i)**2 for ygt_i, yout_i in zip(ygt_vec, yout_vec))
        for ygt_vec, yout_vec in zip(ys, ypred_probs)
    )
    # --- Backward Pass ---
    optimizer.zero_grad() # On remet les gradients à zéro
    loss.backward()      # calculate all gradients
    # --- Mise à jour des poids ---
    optimizer.step()
    print(f"Époque {epoch:2d} | Loss: {loss.data:.4f}")

# Results
print("\n--- Résultats après entraînement ---")
for i, x in enumerate(xs):
    logits = model(x)
    probs = softmax(logits)
    probs_data = [round(p.data, 3) for p in probs]
    pred_class = probs_data.index(max(probs_data))
    real_class = ys[i].index(1)
    
    status = "✅" if pred_class == real_class else "❌"
    print(f"Ex {i} | Probas: {probs_data} | Prédit: {pred_class} | Réel: {real_class} {status}")