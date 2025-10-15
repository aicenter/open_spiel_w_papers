# Revisiting Game Representations: The Hidden Costs of Efficiency in Sequential Decision-making Algorithms


## Abstract

This paper analyses Sequential Bayesian Games (SBGs) as a compact representation of games with imperfect information. We demonstrate that SBGs can significantly reduce representation size compared to Extensive Form Games (EFGs), in domains with private information and public actions. 
We establish formal size comparisons between SBGs and their EFG counterparts, showing possible quadratic reductions in representation size. We describe Public State Counterfactual Regret Minimization (PS-CFR), a game solving algorithm similar to "vanilla" CFR that exploits the structure of SBGs to achieve computational efficiency.
We prove that PS-CFR is never slower than vanilla CFR and can offer asymptotic improvements in some domains. Our empirical evaluations on poker confirm substantial performance gains, showing that PS-CFR significantly outperforms vanilla CFR in practice. 
Additionally, we explore extensions of SBGs that accommodate a wider class of games, characterizing the associated trade-offs in representation size and the modification to the PS-CFR necessary for those extensions.
Our findings suggest that a large part of the success of CFR in popular benchmark domains like poker and Liar's Dice is attributable to their amenability to public state-based implementations. This understanding enables more efficient implementations in new domains with a compatible structure of public and private information.

## Steps to reproduce experiments

Follow the standard OpenSpiel build instructions. 


Then:

```
    $ cd open_spiel_public_state_cfr
    $ ./compile_pscfr.sh
    $ ./install_pscfr.sh
    $ cd open_spiel_public_state_cfr/open_spiel/papers_with_code/public_state_cfr
    $ ./my_main
```
