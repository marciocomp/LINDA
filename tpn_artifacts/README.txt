================================================================================
LINDA - TIMED PETRI NET MODEL AND ANALYSIS REPORTS
================================================================================

The Timed Petri Net of Section IV-E, and the reports PIPE v4.3.0 produced from it. The model is one client's pipeline with n = 3 Compute Workers: 8 places and 10 transitions, each with a single input and a single output place. An arc returns p_Sigma to p_S, closing the net so that the analysis spans successive rounds, and the initial marking M0 holds a single token at p_S.

Tool: PIPE v4.3.0, Platform Independent Petri Net Editor.
Model: linda_tpn_n3.xml


--------------------------------------------------------------------------------
1. PLACES AND TRANSITIONS
--------------------------------------------------------------------------------

The names are the ones Figure 7 and Section IV-E use.

    place      role
    ----------------------------------------------------------------
    p_S        Tier 1 source; holds the token of the initial marking
    p_M        Cluster Manager, ingress of the Tier 2 Compute Pool
    p_H1       first Compute Worker
    p_H2       second Compute Worker
    p_H3       third Compute Worker
    p_D        routing decision point
    p_O        Orchestrator, offloaded partition
    p_Sigma    global barrier

    transition  role
    ----------------------------------------------------------------
    t_P         injection: the Node Agent runs its custody partition
    t_H         forwards the activations into the Compute Worker chain
    t_C1        first worker completes its subset
    t_C2        second worker completes its subset
    t_C3        last worker completes; advances the token to p_D
    t_C0        no additional workers: the token goes straight to p_D
    t_F         strong cluster: the tail finishes locally
    t_W         straggler cluster: offloads its tail over the link
    t_O         the Orchestrator computes the loss and returns gradients
    t_A         federated aggregation: resets the barrier, reinjects at p_S

Section IV-E describes t_A in prose, as the aggregation that fires when M(p_Sigma) = K and resets the barrier; the symbol is introduced here so that every element of the model can be named.


--------------------------------------------------------------------------------
2. WHICH FILE SUPPORTS WHICH CLAIM
--------------------------------------------------------------------------------

    claim in Section IV-E                      file
    ----------------------------------------------------------------------------
    n+5 places and n+7 transitions, n = 3      linda_tpn_n3.xml
    PIPE classifies it as a state machine      classification.html
    one minimal P-invariant, unit weight,      invariant_analysis.html
      giving sum M(p) = 1
    four minimal T-invariants                  invariant_analysis.html
    bounded, safe, no dead markings            state_space_analysis.html
    incidence matrices and initial marking     incidence_and_marking.html
    8 tangible states, 10 state transitions    see section 3 below

The four T-invariants are the four operational configurations of LINDA, and each reproduces M0:

    t_P  t_H t_C1 t_C2 t_C3  t_F        t_A     chained pool, finishing locally
    t_P  t_H t_C1 t_C2 t_C3  t_W t_O    t_A     chained pool, offloading its tail
    t_P  t_C0                t_F        t_A     Cluster Manager alone, local
    t_P  t_C0                t_W t_O    t_A     Cluster Manager alone, offloading


--------------------------------------------------------------------------------
3. THE REACHABILITY GRAPH
--------------------------------------------------------------------------------

The graph is to be generated with the tool itself. PIPE draws it in a window of its own and writes no file: it builds a temporary results.rg, reads it into memory and deletes it, so there is no export from the module and reachability_graph.html records only that the module ran, with its timings.

To obtain it, open linda_tpn_n3.xml in PIPE v4.3.0 and run Reachability/Coverability Graph. With one token in a state machine the graph mirrors the net: each reachable marking is the token sitting in one place, and each arc is one transition firing, which is why the 8 states and the 10 arcs match the 8 places and the 10 transitions of the model.


--------------------------------------------------------------------------------
4. REPRODUCING THE REPORTS
--------------------------------------------------------------------------------

Open linda_tpn_n3.xml in PIPE v4.3.0 and run, from the module tree, Classification, Invariant Analysis, State Space Analysis and Incidence & Marking. Each module window saves its own HTML.

Run the analysis on the untimed net. The structural modules used here read the untimed skeleton, which is what Section IV-E reports; the GSPN modules, which do read the rates, answer a different question and are not part of these claims.
