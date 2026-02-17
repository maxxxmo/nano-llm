import numpy as np
from src.tensor_model.tensor import Tensor # Ton objet Tensor maison
from src.tensor_model.transformer import TransformerBlock # Remplace par le nom de ton fichier

def test_block():
    # 1. Configuration
    B, T, C = 2, 8, 16  # Batch=2, Seq_len=8, d_model=16
    n_heads = 4
    max_seq_len = 10
    
    print(f"Initialisation du bloc (d_model={C}, heads={n_heads})...")
    block = TransformerBlock(d_model=C, n_heads=n_heads, max_seq_len=max_seq_len)
    
    # 2. Création d'une entrée bidon (dummy input)
    # On simule des embeddings de mots
    x_data = np.random.randn(B, T, C).astype(np.float32)
    x = Tensor(x_data)
    
    print(f"Input shape: {x.shape}")
    
    # 3. Passage dans le bloc (Forward)
    try:
        output = block(x)
        print(f"Output shape: {output.shape}")
        
        # Vérification critique : les dimensions doivent être identiques
        assert output.shape == x.shape, f"Erreur de dimension ! Attendu {x.shape}, reçu {output.shape}"
        print("✅ Test de dimension réussi !")
        
    except Exception as e:
        print(f"❌ Erreur pendant le forward : {e}")
        return

    # 4. Vérification des paramètres
    params = block.parameters()
    print(f"Nombre de Tensors de paramètres trouvés : {len(params)}")
    
    # Un bloc contient normalement :
    # Attention: 4 matrices (Q,K,V,O) + MLP: 2 matrices + LN: 2 vecteurs (gamma, beta)
    if len(params) > 0:
        print("✅ Test des paramètres réussi !")
    else:
        print("❌ Attention : Aucun paramètre trouvé !")

if __name__ == "__main__":
    test_block()