# POLICY EVALUATION

## AIM:

To evaluate and compare the performance of two policies using policy evaluation.

## PROBLEM STATEMENT:

* Given a set of states, actions, and transition probabilities, we are given two policies.
* We want to evaluate the performance of the two policies by computing their state-value functions.
* The policy with the higher state-value function is considered to be the better policy.

## POLICY EVALUATION FUNCTION:
```
Developed By: 
Reg.No : 212221240020
```
```python3
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V=np.zeros(len(P))
    while True:
      V=np.zeros(len(P))
      for s in range(len(P)):
        for prob,next_state,reward,done in P[s][pi(s)]:
          V[s]+=prob*(reward+gamma*prev_V[next_state]*(not done))
      if np.max(np.abs(prev_V-V)):
        break
      prev_V=V.copy()
    return V
```

## OUTPUT:

#### Policy 1

![1](https://github.com/saieswar1607/rl-policy-evaluation/assets/93427011/c557ba64-836c-4250-b400-6280a66c61c6)

#### Policy 2

![2](https://github.com/saieswar1607/rl-policy-evaluation/assets/93427011/7f147f43-7c2e-4500-8b7b-698cee0e0b96)

## RESULT:
Thus, the evaluation and comparison of the two policies using policy evaluation has been done successfully and it is found that policy 2 is better.
