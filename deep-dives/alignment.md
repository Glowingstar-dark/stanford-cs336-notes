## Alignment Math: SFT, RLHF/PPO, DPO, GRPO

### Q: The DPO paper claims to optimize the standard RLHF objective without an explicit reward model or reinforcement learning. How is the DPO loss function derived, and what is the key "non-parametric assumption" that makes this transformation from a KL-constrained RL problem to a simple binary classification loss possible?

This is a fantastic question that gets to the heart of DPO's elegance. The derivation connects a complex reinforcement learning objective to a simple supervised loss by re-parameterizing the reward function in terms of the policy itself.

Let's walk through the derivation step-by-step.

**1. The Standard RLHF Objective**

The goal in RLHF is to find a policy $\pi_\theta$ that maximizes the expected reward from a reward model $r_\phi(x,y)$, while not straying too far from an initial reference policy $\pi_{ref}$ (usually the SFT model). This is formulated as a KL-constrained optimization problem:

$$ \max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y|x)} [r(x,y)] - \beta D_{KL}(\pi_\theta(y|x) || \pi_{ref}(y|x)) $$

Here, $\beta$ is a hyperparameter that controls the strength of the KL penalty. A higher $\beta$ means we penalize deviation from $\pi_{ref}$ more heavily.

**2. The Closed-Form Optimal Policy (The "Non-Parametric" Insight)**

The key insight from the DPO paper (and prior work) is that for a *given* reward function $r(x,y)$, this optimization problem has a known closed-form solution for the optimal policy, which we'll call $\pi_r(y|x)$. This solution is found by assuming we can optimize over the space of *all possible distributions* (a "non-parametric" assumption), not just those representable by our specific neural network architecture.

The optimal policy $\pi_r(y|x)$ is:
$$ \pi_r(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right) $$
where $Z(x) = \sum_y \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right)$ is the partition function that ensures the policy probabilities sum to 1 for a given prompt $x$.

*Intuition:* This says the optimal policy is the reference policy, re-weighted by the exponentiated reward. High-reward completions $y$ get their probabilities boosted, and low-reward completions get their probabilities suppressed.

**3. Inverting the Relationship: From Policy to Reward**

The previous step showed how to get the optimal policy from a reward function. DPO's magic is to flip this around. If we have an optimal policy $\pi_r$, what reward function *must have* produced it? We can rearrange the equation above to solve for $r(x,y)$:

$$ \exp\left(\frac{1}{\beta} r(x,y)\right) = \frac{\pi_r(y|x)}{\pi_{ref}(y|x)} Z(x) $$
$$ \frac{1}{\beta} r(x,y) = \log\left(\frac{\pi_r(y|x)}{\pi_{ref}(y|x)}\right) + \log(Z(x)) $$
$$ r(x,y) = \beta \log\left(\frac{\pi_r(y|x)}{\pi_{ref}(y|x)}\right) + \beta \log(Z(x)) $$

This is the crucial step. It establishes a one-to-one mapping: every policy implies a unique reward function (up to a constant that depends only on $x$). This allows us to replace the unknown reward function $r(x,y)$ with a function of our learnable policy $\pi_\theta(y|x)$ and the fixed reference policy $\pi_{ref}(y|x)$.

**4. Modeling Human Preferences with Bradley-Terry**

RLHF data consists of pairwise preferences: for a prompt $x$, a human preferred completion $y_w$ ("winner") over a rejected completion $y_l$ ("loser"). The Bradley-Terry model is a standard way to model the probability of this preference:

$$ p(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l)) = \frac{1}{1 + \exp(-(r(x, y_w) - r(x, y_l)))} $$

This says the probability of preferring $y_w$ increases as the difference in their rewards grows.

**5. Substituting the Implied Reward into the Preference Model**

Now, we substitute our policy-defined reward from Step 3 into the Bradley-Terry model. Let's look at the difference in rewards:

$$ r(x, y_w) - r(x, y_l) = \left( \beta \log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} + \beta \log Z(x) \right) - \left( \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} + \beta \log Z(x) \right) $$
$$ r(x, y_w) - r(x, y_l) = \beta \left( \log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) $$

Notice that the unknown partition function $Z(x)$ conveniently cancels out! Let's define the implicit reward estimate for a policy $\pi_\theta$ as $\hat{r}_\theta(x,y) = \beta \log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$.

The preference probability is now:
$$ p(y_w \succ y_l | x) = \sigma(\hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l)) $$

**6. The Final DPO Loss**

The goal is to find the policy $\pi_\theta$ that maximizes the likelihood of the observed human preferences. This is a standard maximum likelihood estimation problem. We want to maximize the log-probability of the data, which is equivalent to minimizing the negative log-likelihood:

$$ \mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma(\hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l)) \right] $$

Substituting the definition of $\hat{r}_\theta$:
$$ \mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right) \right] $$

This is exactly the DPO loss. It's a simple binary cross-entropy loss where we are classifying which response is preferred. The "logits" for this classification are the difference in the implicit rewards.

**Implementation and Pitfalls**

In practice, the implementation is straightforward and resembles supervised fine-tuning.

```python
def dpo_loss(policy_logits_w, policy_logits_l,
             ref_logits_w, ref_logits_l,
             beta=0.1):
    """
    Computes the DPO loss for a batch of winning and losing responses.
    Logits are assumed to be per-token, shape [batch_size, seq_len, vocab_size].
    """
    # Get log-probabilities of the actual sequences
    # These are sums over the sequence length
    pi_log_probs_w = get_sequence_log_probs(policy_logits_w, y_w) # shape: [batch_size]
    pi_log_probs_l = get_sequence_log_probs(policy_logits_l, y_l) # shape: [batch_size]

    ref_log_probs_w = get_sequence_log_probs(ref_logits_w, y_w) # shape: [batch_size]
    ref_log_probs_l = get_sequence_log_probs(ref_logits_l, y_l) # shape: [batch_size]

    # Calculate implicit rewards
    pi_ratio_w = pi_log_probs_w - ref_log_probs_w
    pi_ratio_l = pi_log_probs_l - ref_log_probs_l

    # The "logits" for the binary classification
    logits = pi_ratio_w - pi_ratio_l

    # Binary cross-entropy loss with a target of 1 (we want to prefer y_w)
    loss = -F.logsigmoid(beta * logits).mean()
    return loss
```

*   **Numerical Stability:** The log-probability ratios can be large. However, since the loss is a `logsigmoid`, it's generally stable.
*   **Reference Model:** The reference model `pi_ref` is kept frozen during training. This requires having two models in memory (the policy being trained and the reference), but only one requires gradients.
*   **Hyperparameter $\beta$:** This parameter is crucial. It acts as an inverse temperature. If $\beta$ is too high, the gradients can become very large, leading to instability. If it's too low, the learning signal is weak. Typical values are around 0.1 to 0.5.

---

### Q: The GRPO algorithm simplifies PPO by replacing the learned value function with a z-score normalization of rewards. Why, mathematically, does this z-scoring introduce bias into the policy gradient estimate, and what are the practical consequences of this bias?

This is a subtle but critical point that highlights the trade-offs between algorithmic simplicity and theoretical correctness. While GRPO's z-scoring is an effective heuristic, it violates a core assumption of policy gradient methods, leading to predictable and sometimes undesirable side effects.

**1. The Role of a "Valid" Baseline in Policy Gradients**

Let's start with the Policy Gradient Theorem. The gradient of the expected reward $J(\theta)$ is:
$$ \nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(\tau) \cdot R(\tau) \right] $$
where $\tau$ is a trajectory (a full response in our case) and $R(\tau)$ is its total reward.

To reduce the high variance of this estimate, we introduce a baseline $b(s)$ that depends on the state (prompt) but not the action (response). The gradient becomes:
$$ \nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(\tau) \cdot (R(\tau) - b(s)) \right] $$
This estimate remains **unbiased** (i.e., its expectation is still the true gradient) if and only if the baseline term has an expectation of zero:
$$ \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(\tau) \cdot b(s) \right] = 0 $$

Let's prove this condition. We can rewrite the expectation as an integral over trajectories:
$$ \int \pi_\theta(\tau|s) p(s) \left( \nabla_\theta \log \pi_\theta(\tau|s) \cdot b(s) \right) d\tau ds $$
Using the log-derivative trick ($\nabla_\theta \pi_\theta = \pi_\theta \nabla_\theta \log \pi_\theta$):
$$ = \int p(s) b(s) \left( \nabla_\theta \pi_\theta(\tau|s) \right) d\tau ds $$
$$ = \int p(s) b(s) \nabla_\theta \left( \int \pi_\theta(\tau|s) d\tau \right) ds $$
Since $\int \pi_\theta(\tau|s) d\tau = 1$ (probabilities sum to 1), its gradient is zero: $\nabla_\theta(1) = 0$.
$$ = \int p(s) b(s) \cdot 0 \cdot ds = 0 $$
This holds as long as $b(s)$ does not depend on the trajectory $\tau$ (or the actions within it), allowing us to pull it out of the integral over $\tau$.

**2. How GRPO Violates the Baseline Condition**

GRPO proposes an "advantage" estimate for a response $i$ within a group $G$ of responses to the same prompt:
$$ \hat{A}_i = \frac{r_i - \text{mean}(r_1, \dots, r_{|G|})}{\text{std}(r_1, \dots, r_{|G|}) + \epsilon} $$
Here, the baseline for response $i$ is not just the mean, but a function of the entire group's rewards. Let's focus on the denominator: $b_i = \text{std}(r_1, \dots, r_{|G|})$.

The standard deviation is calculated from the rewards of *all* responses $\{r_1, \dots, r_{|G|}\}$ in the group. Each reward $r_j$ depends on the trajectory $\tau_j$ sampled to generate it. Therefore, the denominator $\text{std}(\dots)$ is a function of the entire set of sampled trajectories $\{\tau_1, \dots, \tau_{|G|}\}$.

When we compute the gradient for a single trajectory $\tau_i$, the term $\nabla_\theta \log \pi_\theta(\tau_i)$ is multiplied by an "advantage" whose denominator depends on $\tau_j$ for all $j \neq i$. This coupling means the baseline is no longer independent of the actions, violating the condition for an unbiased estimate. The gradient for one sample is being scaled by a factor that depends on other random samples in the batch, introducing bias.

**3. Practical Consequences of the Biased Gradient**

This theoretical flaw has real, practical consequences during training:

*   **Length-Based Reward Hacking:** The most significant issue is a bias related to response length. The policy gradient update for response $i$ is roughly proportional to $\frac{r_i - \bar{r}}{\text{std}}$.
    *   **If an answer is wrong ($r_i < \bar{r}$):** The model is incentivized to make the response *very long*. A longer response might not improve the reward $r_i$, but if it's generated with low probability, it can increase the variance of rewards in the group, thus increasing the denominator `std`. A larger denominator shrinks the negative advantage term, reducing the penalty. This encourages the model to "BS" at length when it doesn't know the answer.
    *   **If an answer is right ($r_i > \bar{r}$):** The model is incentivized to make the response as *short as possible*. A shorter correct answer is often generated with higher probability, potentially reducing group variance and maximizing the positive advantage.

    This explains the phenomenon observed in models like DeepSeek-Coder, where RL training with GRPO leads to a dramatic and often unbounded increase in the length of Chain-of-Thought (CoT) reasoning, which may be an artifact of the biased objective rather than a sign of "deeper thinking."

*   **Distorted Focus:** The `std` term in the denominator causes the algorithm to place more weight on prompts where the model's performance is inconsistent (high variance) and less weight on prompts where the model is either consistently right or consistently wrong (low variance). This can slow down learning on "easy" or "hopeless" problems.

**In summary:** GRPO's z-scoring is a clever, simple heuristic that provides a powerful adaptive baseline without needing a separate critic network. However, its theoretical flaw introduces a bias that can lead to undesirable behaviors like pathological length increases. Recent work has proposed fixes, such as removing the standard deviation term (recovering a simple baselined policy gradient) or applying length normalization to counteract the bias.

```python
# Pseudocode illustrating the coupling in GRPO
def grpo_advantage(rewards_in_group):
    # rewards_in_group is a tensor of rewards for N responses to the same prompt
    mean_reward = rewards_in_group.mean()
    std_reward = rewards_in_group.std() # <-- Problem here!

    # The std_reward depends on ALL rewards in the group.
    # When calculating the gradient for one response, its "advantage"
    # is scaled by a term that depends on other random responses.
    advantages = (rewards_in_group - mean_reward) / (std_reward + 1e-8)
    return advantages

# In the training loop...
# for each prompt:
#   responses = model.generate(prompt, num_responses=N)
#   rewards = compute_rewards(responses)
#   advantages = grpo_advantage(rewards) # Biased!
#   for i in 1..N:
#     log_prob_i = compute_log_prob(responses[i])
#     loss += -log_prob_i * advantages[i].detach() # .detach() doesn't fix the bias
```

---

### Q: The KL-divergence term appears in both the overall RLHF objective (e.g., DPO) and within the inner loop of policy gradient algorithms like PPO. What are the distinct roles of the KL term in these two contexts, and how does the choice of reference policy ($\pi_{ref}$ vs. $\pi_{old}$) reflect these different goals?

This is an excellent question that untangles a common source of confusion. The KL-divergence term is a multi-purpose tool in alignment, serving two distinct functions: one as a **global behavioral constraint** and the other as a **local update stabilizer**.

#### Role 1: Global Behavioral Constraint

*   **Context:** This is the KL term in the main RLHF objective, which is optimized over the entire training run.
    $$ \max_{\pi_\theta} \mathbb{E}[r(x,y)] - \beta D_{KL}(\pi_\theta(y|x) || \pi_{ref}(y|x)) $$
*   **Goal:** To prevent the final, optimized policy $\pi_\theta$ from deviating too far from a trusted, high-quality initial policy. Its purpose is to preserve general capabilities and prevent "reward hacking" or "mode collapse." The model should get better at the specific task (e.g., math problems) without forgetting how to write coherent English, follow general instructions, or maintain its safety profile.
*   **Reference Policy ($\pi_{ref}$):** This is typically the **Supervised Fine-Tuned (SFT) model**. The SFT model is chosen because it has already been trained on a diverse set of high-quality demonstrations and represents a good, general-purpose starting point. It embodies the desired style and broad capabilities we don't want to lose. This $\pi_{ref}$ is **fixed** throughout the entire RLHF training process.
*   **Timescale:** Global, across the entire training procedure.
*   **Pitfall:** If the KL coefficient $\beta$ is too low, the policy may over-optimize for the reward signal, leading to narrow, brittle, or nonsensical behavior that achieves a high score but is not generally useful (e.g., outputting only "The answer is 42." to every math problem). If $\beta$ is too high, the policy will be too constrained and unable to learn from the reward signal, effectively never improving beyond the SFT model.

#### Role 2: Local Update Stabilizer

*   **Context:** This KL term appears within the inner loop of on-policy RL algorithms like PPO and TRPO. These algorithms are based on importance sampling. They generate a batch of experience (rollouts) using a policy $\pi_{old}$, and then use that data to perform one or more gradient updates on the current policy $\pi_\theta$.
*   **Goal:** To ensure the stability of the gradient updates. The importance sampling ratio, $w_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}$, is used to correct for the fact that the data came from a different policy. If $\pi_\theta$ becomes too different from $\pi_{old}$, this ratio can become very large or small, leading to extremely high variance in the gradient estimates and causing the training to diverge. The KL constraint keeps $\pi_\theta$ inside a "trust region" around $\pi_{old}$ where the importance sampling estimate is reliable.
*   **Reference Policy ($\pi_{old}$):** This is the **policy from the previous iteration**—the one used to generate the current batch of data. Unlike the global $\pi_{ref}$, this $\pi_{old}$ is **updated frequently** (typically after every batch or set of gradient steps). It's a moving target that tracks the learning progress.
*   **Timescale:** Local, within a single batch of updates.
*   **Implementation:**
    *   **TRPO** enforces this as a hard constraint: $D_{KL}(\pi_\theta || \pi_{old}) \le \delta$.
    *   **PPO** uses a simpler "clipping" mechanism on the probability ratio as a proxy for the KL constraint, which is computationally cheaper and often works just as well.
    $$ L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min(w_t \hat{A}_t, \text{clip}(w_t, 1-\epsilon, 1+\epsilon) \hat{A}_t) \right] $$

#### Summary Table

| Feature               | Role 1: Global Behavioral Constraint                               | Role 2: Local Update Stabilizer                                    |
| --------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| **Purpose**           | Prevent catastrophic forgetting, preserve general capabilities.    | Ensure stable gradient updates via importance sampling.            |
| **Algorithm Context** | Overall RLHF objective (DPO, PPO's full objective).                | Inner loop of on-policy algorithms (PPO, TRPO, GRPO).              |
| **Reference Policy**  | $\pi_{ref}$ = SFT Model (fixed).                                   | $\pi_{old}$ = Policy from previous iteration (frequently updated). |
| **Timescale**         | Global (entire training run).                                      | Local (per batch of updates).                                      |
| **Effect**            | Keeps the final model "well-behaved" and generalist.               | Prevents training from diverging due to high-variance gradients.   |

In the full PPO objective used by InstructGPT, both terms are actually present: one for the reward, one for the global KL penalty against the SFT model, and the clipping mechanism for local stabilization. DPO cleverly sidesteps the need for local stabilization by being an offline, supervised-style algorithm, so it only needs to consider the global KL constraint, which is baked directly into its loss function.