import numpy as np
from src.tensor_model.nn import Linear, ReLU, LayerNorm 
from src.tensor_model.tensor import Tensor
def test_mlp():
    print("--- Test du MLP et de la propagation des gradients ---")
    
    # 1. Initialisation
    batch_size = 4
    in_features = 8
    out_features = 3
    
    x = Tensor(np.random.randn(batch_size, in_features))
    model = Linear(in_features, out_features)
    
    # 2. Forward pass
    # Attention : Ton Linear fait W @ x + b. 
    # Si x est (batch, in) et W est (in, out), alors W @ x ne marche pas !
    # Il faudrait x @ W + b.
    try:
        logits = model(x)
        print(f"Forward success: Output shape {logits.data.shape}")
    except ValueError as e:
        print(f"❌ Erreur de dimension : {e}")
        return

    # 3. Backward pass
    logits.backward()
    
    # 4. Vérification des paramètres
    params = model.parameters()
    print(f"Nombre de paramètres : {len(params)}") # Devrait être 2 (W et b)
    
    for i, p in enumerate(params):
        has_grad = np.any(p.grad != 0)
        print(f"Paramètre {i} ({p.data.shape}) a reçu un gradient : {has_grad}")
    
    # 5. Test du zero_grad
    model.zero_grad()
    grads_cleaned = all(np.all(p.grad == 0) for p in model.parameters())
    print(f"Zero Grad fonctionne : {grads_cleaned}")

if __name__ == "__main__":
    test_mlp()