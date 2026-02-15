import numpy as np
from src.tensor_model.tensor import Tensor
from src.tensor_model.mlp import MLP

def test_mlp_complex():
    print("--- Test du bloc MLP (Forward + Backward) ---")
    
    # 1. Init
    n_embd = 4
    batch_size = 2
    model = MLP(n_embd)
    
    # Input avec quelques valeurs négatives pour tester la ReLU
    x_data = np.array([
        [1.0, -0.5, 2.0, 0.0],
        [-1.0, 1.5, -0.5, 1.0]
    ])
    x = Tensor(x_data)
    
    # 2. Forward
    output = model(x)
    print(f"Sortie shape: {output.data.shape}") # Doit être (2, 4)
    
    # 3. Backward
    output.backward()
    
    # 4. Vérification des gradients
    params = model.parameters()
    # On a 2 couches Linear, chaque couche a W et b -> 4 paramètres au total
    print(f"Nombre total de paramètres récupérés: {len(params)}")
    
    all_grads_ok = True
    for i, p in enumerate(params):
        grad_norm = np.linalg.norm(p.grad)
        if grad_norm == 0:
            print(f"⚠️ Paramètre {i} a un gradient nul (possible mais à vérifier)")
            all_grads_ok = False
        else:
            print(f"✅ Paramètre {i} ({p.data.shape}) gradient norm: {grad_norm:.6f}")
            
    if all_grads_ok:
        print("--- Résultat: Le flux de gradient est complet ! ---")

if __name__ == "__main__":
    test_mlp_complex()