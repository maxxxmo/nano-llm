\# Creation of a LLM using Transformers from Scratch



\## Plan (in french here)



\- Neurone

\- Réseau de neurone / MLP

\- Tokenizer

\- Embedding

\- Self Attention

\- Transformer

\- Trainer

\- Inference





## 1 Neuron class
(easiest one)

For the neuron class creation we need the following parameters:
- weights
- bias

I have y = wx + b 
where :
x --> input
w --> weights
b --> bias
y --> output of the neuron


![source medium (Matthew Roos)](./images/neuron.png)

## Issue
Then i need to implement activation functions but it makes me realize i have no way of following operation for backpropagation so i need to creat a 'Automatic differenciation / Tape based autodiff' like in pytorch or the value class in Karpathy's videos


# 1 Automatic differenciation
***(Using Karpathy's structure but i make it up to date by adding exponential, tanh, log, sigmoid, leaky_relu), Also like pytorch do i need a with torch.no_grad mode,  and optimizer()***

zeroing of gradients is already implemented in his code --> usefull for zeroing between epochs.

### Explication Backpropagation

Les définitions d'opérations consistent à refaire les opérations en utilisant la classe `Value` qui contient l'historique pour chaque valeur.

![alt text](images/back.png)

Ici on veut le gradient de **L** la sortie. Le gradient de la sortie par rapport à elle-même est **1**.  
On veut ensuite le gradient de **d** $\rightarrow$ $\frac{dL}{dd}$. 

Or $L = d \cdot f$ donc $\frac{dL}{dd} = f$.  
Donc le gradient de **d** est **-2**.

![alt text](/images/back2.png)

On a maintenant **d** de l'exemple précédent qui est le résultat de l'opération $d = c + e$.  
On cherche le gradient de **c**. Donc on veut $\frac{dL}{dc}$ et on connaît déjà $\frac{dL}{dd}$, il manque $\frac{dd}{dc}$.

Comme $d = c + e$, alors $\frac{dd}{dc} = 1$.

#### Chain Rule :
$$\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$$

Donc pour obtenir le résultat final il suffit de multiplier tous les gradients intermédiaires.  
Donc le gradient de **c** $= 1 \cdot -2 = -2$.

### Tri

Maintenant on veut backpropagate mais il faut que tout les nodes avant soient traités avant de faire les suivants. On utilise un tri topologque. Puis calculer les gradients en suivant ce tri en partant de la sortie.


## 2 Neuron class


# Next steps
 0. passer en numpy le core pour efficacité computationnelle    

0.5 batching

1. la tokenization et embedding

1.2 Fonction de pertes (cross entropy)

2. les classes d'attention (head puis multi avec masquage)

3. la classe transformer (avec residual connections et layer normalization)

4. Fonction generate 



## Change from Value to tensored values
We were calculating each value but its to slow for a LLM. SO we will now use tensor
Value --> Tensor
Neuron --> Linear
Layer --> Linear
loops for neurons --> vectorisation






## Sources

- What i remember from Andrej Karpathy videos on the subject

- also https://medium.com/@prxdyu/simple-neural-network-in-python-from-scratch-2814443a3050

- https://huggingface.co/blog/andmholm/what-is-automatic-differentiation


## What i learnt in the project

- pip freeze > requirements.txt   --> take installed versions and write them in requirements.txt
