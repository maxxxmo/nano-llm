import numpy as np
import torch
from src.tensor_model.tensor import Tensor 

def verify_step(name, custom_tensor, torch_tensor):
    """Vérifie la data et le gradient entre ton moteur et PyTorch."""
    # 1. Vérification de la donnée (Forward)
    data_match = np.allclose(custom_tensor.data, torch_tensor.detach().numpy(), atol=1e-6)
    
    # 2. Vérification du gradient (Backward)
    # On compare si les deux existent
    custom_grad = custom_tensor.grad
    torch_grad = torch_tensor.grad.numpy() if torch_tensor.grad is not None else None
    
    grad_match = True
    if torch_grad is not None:
        grad_match = np.allclose(custom_grad, torch_grad, atol=1e-6)
    
    status = "✅ PASS" if (data_match and grad_match) else "❌ FAIL"
    print(f"[{status}] {name}")
    
    if not data_match:
        print(f"    Data mismatch! \n    Custom: {custom_tensor.data} \n    Torch: {torch_tensor.detach().numpy()}")
    if not grad_match:
        print(f"    Grad mismatch! \n    Custom: {custom_grad} \n    Torch: {torch_grad}")

# --- SCÉNARIOS DE TEST ---

print("--- Démarrage des tests unitaires ---")

# Test 1: Opérations de base & Broadcasting
x_val, y_val = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
x = Tensor(x_val); x_pt = torch.tensor(x_val, requires_grad=True, dtype=torch.float64)
y = Tensor(y_val); y_pt = torch.tensor(y_val, requires_grad=True, dtype=torch.float64)

z = (x * y) + x**2
z_pt = (x_pt * y_pt) + x_pt**2

z.backward()
z_pt.sum().backward() # .sum() car ton backward initialise à 1 sur toute la shape
verify_step("Basic Ops (add, mul, pow)", x, x_pt)

# Test 2: Matmul & Shapes
a_val = np.random.randn(2, 3)
b_val = np.random.randn(3, 2)
a = Tensor(a_val); a_pt = torch.tensor(a_val, requires_grad=True)
b = Tensor(b_val); b_pt = torch.tensor(b_val, requires_grad=True)

c = a @ b
c_pt = a_pt @ b_pt

c.backward()
c_pt.backward(torch.ones_like(c_pt))
verify_step("Matmul (2x3 @ 3x2)", a, a_pt)
verify_step("Matmul Grad B", b, b_pt)

# Test 3: Activation Functions (GELU)
g_val = [-1.0, 0.0, 1.0, 2.0]
g = Tensor(g_val); g_pt = torch.tensor(g_val, requires_grad=True)
out_g = g.gelu()
out_g_pt = torch.nn.functional.gelu(g_pt, approximate='tanh') # Ton code utilise l'approx tanh

out_g.backward()
out_g_pt.backward(torch.ones_like(out_g_pt))
verify_step("GELU Activation", g, g_pt)