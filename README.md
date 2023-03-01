# Tsunami

Github du projet de recherche Tsunami : résoudre à l'aide de réseaux neuronaux les équations de propagation d'un tsunami.

# Contexte :
CentraleSupélec : Pôle 005 projet formation à la recherche

# Encadrant :
- Frédéric Magoulès

# Membres du projet :
- Rosenberger Julien
- Oliveira Lima Lawson
- Paun Théodore
- Antier Esteban

# Structure du repository
```
├───doc  
│   ├───Bibliography              -> Reference articles
│   └───polynomails_doc           -> Research on polynomials for certain AI methods  
├───data
│   ├───arcachon_bathymetry       -> Medium resolution dataset
│   ├───arcachon_plus_bathymetry  -> High resolution dataset
│   └───Atlantic_bathymetry       -> Low resolution dataset  
├───manuscript                    -> Reports
├───processed_data
│   ├───Mesh                      -> Mesh for Arcachon basin
│   └───Boundary                  -> Mesh for the boundary
└───src
    ├───Approaches                -> Tests based on Lagaris' studies
    ├───Deprecated                -> Old code using polynomials for boundary conditions
    ├───Neural_Networks           -> First tests using PINN with Tf, Torch and Jax (time independent)
    ├───dist                      -> Distance functions
    └───Simulations               -> Solving PDEs with Jax
```
