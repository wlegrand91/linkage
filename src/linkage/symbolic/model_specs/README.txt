README file for the generic binding model implementation.

The generic binding model processes a .txt file of the following format:

equilibria:
    C + E->EC; KE
    A -> I; KI
    A + C -> AC1; K1
    AC1 + C -> AC2; K2
    AC2 + C -> AC3; K3
    AC3 + C -> AC4; K4

species:
    ET = E + EC
    AT = I + A + 2*AC1 + AC2 + 2*AC3 + AC4
    CT = C + EC + 2*AC1 + 2*AC2 + 6*AC3 + 4*AC4

In it's current implementation, the generic model will attempt using a sympy algorithm to solve for C in terms of CT and
other macrospecies concentrations (i.e. ET, CT) and the equilibrium constants defined to the right of the semicolon in the 
equilibria portion of the .txt file. It is also possible to assign a python string within the .ipynb file as the model, 
which may be useful for on-the-fly modifications. 

Two important considerations:
-Not all equilibria/species models are able to be converted into a functional binding polynomial, and it is worth checking 
a simple implementation of your model by hand (i.e. with degeneracy of 1 for each microspecies)
-The rational equation final_ct that the generic model produces is generally slower to iterate than a hand-written polynomial. 
If computation is a limiting factor, it may be worth writing out the rational equation (accessible via gm._bm.print_summary() for
an instantiation of global model called gm) and manipulating it into a polynomial equation. Even if this is the case, much of the
algebraic legwork will be done already.
