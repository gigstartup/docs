Algorithmic Valuation Appendix
Algorithmic Valuation is around measuring improvements - did you decrease the delta around what you wanted to achieve, and where you started.
Allotments are used as a means to help demonstrate to people valuation in a transactional manner.  However, the same calculation can also be done using mathematics, without the need for Allotments, using the formula:
|Qˆ (s, a ) − Q (s, a )| = |( r + γ max a ′ Qˆ ( s ′ , a ′)) − ( r + γ max a ′ Q ( s ′ , a ′ )| = γ| max a ′ Qˆ ( s ′ , a ′ ) − max a ′ Q ( s ′ , a ′ )| ≤ γ max a ′ |Qˆ ( s ′ , a ′ ) − Q ( s ′ , a ′ )| ≤ γ · γ i − 1 ∆ 0.
Obtainable outcome with measurable actions and observable states to meet a purpose or objective. By taking the Quality or Advantage (highest probable Polyform or Action) and discounting the maximum reward from all observed states and measured actions necessary to decrease the delta from desired outcome to existing states and actions we achieve the constrained optimal valuation. We can now provide a path to decrease loss, and maximize rewards. We solve for potential. 
Definitions:
V π(s, a) = Polyform, or V π(s) = a if Deterministic
Where:
V = Value
Qˆ = Quality
π = Policy
s = State
a = Action
γ  = Discounted
r = Reward

NOTES: (work in progress)

Example:
The goal of the algorithm is to determine the state, s, where the delta is largest between the desired new policy, result (q target) and discounted reward γ(r).
Certain vs Uncertain Outcomes can be split into Deterministic Outcomes and Stochastic Outcomes. 
Certain Outcomes have a desired Result (r) and state (s) and the action (a) is already determined.
“Want to launch a company” = r, “We have the know-how, money, url, the form filled out and the time to launch an LLC  without anything inhibiting us” = s. Therefore the action I need to take next is to submit the form = (a). 
Uncertain Outcomes have r(s,a) = P(s|a)
“I want to succeed” = r, “I have a company, target market, traction, technology, team, etc.” = s, Therefore I may or may not succeed, but can try to determine probability P.
Depict the sum of rewards, not a single objective optimized. Maximize the entropy of the policy. (what is the best representation of the state?)
Math:
Discounted Reward:
r = γ(r)
π new, q-target, γ(r) =  [Qθ(s, a) − log πφ(a|s)]
Advantage:
A = Q(s,a) - V(s)
Advantage Estimate:
A = r - V(s)
Soft Value Function trained to minimize squared residual error:
JV (ψ) = Est∼D h 1 2 Vψ(st) − Eat∼πφ [Qθ(st, at) − log πφ(at|st)]2 i
Unbiased Estimator:
 ∇ˆ ψJV (ψ) = ∇ψVψ(st) (Vψ(st) − Qθ(st, at) + log πφ(at|st)
Minimize soft Bellman residual:
 JQ(θ) = E(st,at)∼D  1 2  Qθ(st, at) − Qˆ(st, at) 2  
with:
 Qˆ(st, at) = r(st, at) + γ Est+1∼p Vψ¯(st+1) 
Optimized with Stochastic Gradients
∇ˆ θJQ(θ) = ∇θQθ(at, st) Qθ(st, at) − r(st, at) − γVψ¯(st+1)  
Code:
def success(self,global_model,rollout,sess,bootstrap_value):
    rollout = np.array(rollout)
    observations = rollout[:,0]
    actions = rollout[:,1]
    rewards = rollout[:,2]
    next_observations = rollout[:,5]

#take rewards and values from the rollout to generate the advantage and d    #discounted returns.
    self.rewards_plus = np.asarray(rewards.tolist()) + [bootstrap_value])
    discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
    self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
    advantages = rewards + gamma + self.value_plus[1:] - self.value_plus[:-1]
    advantages = discount(advantages,gamma)

#ann init
More Code:

#### param tuning, these can be altered for a problem. Gamma and Soft_Tau can be replaced. Soft_Tau is the activation function. Probability distributions and standard deviations (microcompany purpose) help determine the right choice. ####


def stoch_q_update(batch_size,
    		gamma=0.99,
    		mean_lambda=1e-3,
    		std_lambda=1e-3,
    		z_lambda=0.0,
    		soft_tau=1e-2,
   	           ):
#### Rest of the function - should be indented appropriately ####

#dim, pan, allotment, next_state, purpose_met
state, action, reward, next_state, done = replay_buffer.sample(batch_size)

#A polyform can serve the same purpose as a FloatTensor in this case
state = torch.FloatTensor(state).to(device)
next_state = torch.FloatTensor(next_state).to(device)
action = torch.FloatTensor(action).to(device)
reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)
Env = NormalizedEnv(du.make(pan, dim))

action_pan = env.action_space.shape[0]
state_dim = env.observation_space.shape[0
hidden_dim = 256

value_lattice = ValueNetwork(state_dim, hidden_dim).to(device)
target_value_lattice = ValueNetwork(state_dim, hidden_dim).to(device)

stoch_q_lattice = StochQNetwork(state_dim, action_pan, hidden_dim).to(device)
policy_lattice = PolicyNetwork(state_dim, action_pan, hidden_dim).to(device)

for target_param, param in zip(target_value_lattice.parameters(), value_lattice.parameters()):
    target_param.data.copy_(param.data)
value_criterion = nn.MSELoss()
stoch_q_criterion = nn.MSELoss()


#lr = learning rate
value_lr = 3e-4
stoch_q_lr = 3e-4
policy_lr = 3e-4

value_optimizer = optim.Adam(value_lattice.parameters(), lr=value_lr)
stoch_q_optimizer = optim.Adam(stoch_q_lattice.parameters(), lr=stoch_q_lr)
policy_optimizer = optim.Adam(policy_lattice.parameters(), lr=value_lr)
Example:
#### TODO - Add SAC sota and mult part/entities ####
Missing: discounted reward
Probability distribution
Account for exceptions that will come up in people’s minds
