import numpy as np
from src.tensor_model.tensor import Tensor
from src.tensor_model.mlp import MLP
from src.tensor_model.dataloader import DataLoader 

def test_mlp_batch_integration():
    print("--- Test Intégration MLP + DataLoader ---")
    
    # 1. Configuration
    n_samples = 10
    n_embd = 4
    batch_size = 3
    model = MLP(n_embd)
    
    # Données fictives (X: features, y: cibles pour calculer une perte)
    X_data = np.random.randn(n_samples, n_embd)
    y_data = np.random.randn(n_samples, n_embd)
    
    loader = DataLoader(X_data, y_data, batch_size=batch_size, shuffle=True)
    
    # 2. Simulation d'une "Époque" d'entraînement
    print(f"Lancement sur {n_samples} échantillons avec batch_size={batch_size}\n")
    
    for i, (x_batch, y_batch) in enumerate(loader):
        # On remet les gradients à zéro pour chaque nouveau batch !
        for p in model.parameters():
            p.grad = np.zeros_like(p.data)
            
        # Forward
        output = model(x_batch)
        
        # Calcul d'une loss simple (MSE) pour déclencher le backward
        # On suppose que tu as implémenté les opérations nécessaires
        diff = output - y_batch
        loss = (diff ** 2).mean() 
        
        # Backward
        loss.backward()
        
        print(f"Batch {i+1} | Loss: {loss.data:.6f} | Shape: {x_batch.data.shape}")
        
        # Vérification sur le premier paramètre pour l'exemple
        first_param = model.parameters()[0]
        grad_norm = np.linalg.norm(first_param.grad)
        
        assert first_param.grad.shape == first_param.data.shape
        assert grad_norm > 0, f"Le gradient est nul au batch {i+1}"
        
        print(f"   ∟ Gradient norm (Param 0): {grad_norm:.6f} ✅")

    print("\n--- Résultat: L'intégration Batch + MLP fonctionne ! ---")

if __name__ == "__main__":
    test_mlp_batch_integration()