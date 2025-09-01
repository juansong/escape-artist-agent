from experiments.train_mc import train_mc
from experiments.evaluate import evaluate
from environment.escape_env import EscapeEnv
from agent.monte_carlo import MCAgent
from agent.importance_sampling import OffPolicyMCAgent

def compare_methods():
    env = EscapeEnv(grid_size=5)

    # First-Visit MC
    agent_fv = MCAgent(env.action_space)
    train_mc_agent(agent_fv, env)
    print("Evaluating First-Visit MC")
    evaluate(env, agent_fv)

    # Every-Visit MC
    agent_ev = MCAgent(env.action_space)
    train_mc_agent(agent_ev, env, every_visit=True)
    print("Evaluating Every-Visit MC")
    evaluate(env, agent_ev)

    # Off-policy MC
    off_agent = OffPolicyMCAgent(env.action_space)
    # Example: use episodes collected from behavior policy
    # off_policy_update(off_agent, episodes)
    print("Evaluating Off-policy MC")
    evaluate(env, off_agent)

if __name__ == "__main__":
    compare_methods()
