
import random
import numpy as np
from src.model.mlp import MLP
from src.model.core import Value

def test_debug():
    print("=== Phase 1: Test de l'initialisation ===")
    dtype_test = 'float32'
    model = MLP(3, [4, 1], dtype=dtype_test)
    
    # Vérification du type des paramètres
    all_params = model.parameters()
    types_corrects = all(p.dtype == dtype_test for p in all_params)
    print(f"[*] Tous les paramètres sont en {dtype_test}: {types_corrects}")
    
    # Vérification des data types réels (numpy)
    sample_p = all_params[0]
    print(f"[*] Type réel de data: {type(sample_p.data)} | Valeur: {sample_p.data}")

    print("\n=== Phase 2: Propagation (Forward Pass) ===")
    x = [2.0, 3.0, -1.0]
    
    # On teste chaque couche une par une pour isoler le NaN
    current_input = x
    for i, layer in enumerate(model.layers):
        try:
            output = layer(current_input)
            # Si output est une liste (cas des couches cachées), on vérifie chaque élément
            if isinstance(output, list):
                has_nan = any(np.isnan(v.data) for v in output)
                current_dtype = output[0].dtype
            else:
                has_nan = np.isnan(output.data)
                current_dtype = output.dtype
            
            print(f"[Layer {i}] Sortie OK | Nan détecté: {has_nan} | Dtype: {current_dtype}")
            
            if has_nan:
                print(f" /!\ Alerte : Explosion détectée dans la couche {i}")
                break
            current_input = output
        except Exception as e:
            print(f"[Layer {i}] Erreur fatale: {e}")
            break

    print("\n=== Phase 3: Test de la Loss et Backward ===")
    try:
        y_pred = model(x)
        y_target = 1.0
        loss = (y_pred - y_target)**2 # Vérifie si tu as implémenté __sub__ ou si c'est (y_pred + (-y_target))
        print(f"[*] Loss: {loss.data}")
        
        model.zero_grad()
        loss.backward()
        print(f"[*] Backward terminé sans crash.")
        
        # Vérification si un gradient est NaN
        nan_grads = any(np.isnan(p.grad) for p in model.parameters())
        print(f"[*] Gradients NaN détectés: {nan_grads}")
        
    except Exception as e:
        print(f"[!] Erreur pendant la loss/backward: {e}")

if __name__ == "__main__":
    test_debug()