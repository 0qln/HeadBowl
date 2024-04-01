using System.Numerics;

namespace HeadBowl.ReinforcementLearning;

/// <summary>
/// https://github.com/0qln/Obsidian/blob/main/Main/Coding/Q%20Learning.md#monte-carlo-learning
/// </summary>
/// <typeparam name="TPrecision"></typeparam>
/// <typeparam name="TState"></typeparam>
/// <typeparam name="TAction"></typeparam>
public static class MonteCarloQLearning<TPrecision>
    where TPrecision : struct, ISubtractionOperators<TPrecision, TPrecision, TPrecision>
{
    public static void Update<TState, TAction>(IAgent<TPrecision, TState, TAction> agent, TPrecision totalReward, TState state, TAction action, int numActions)
        where TState : IEnviroment<TAction>
    {
        TPrecision[] output = [agent.Q(state, action)];
        TPrecision[] input = agent.GetInputs(state, action);
        TPrecision[] excpected = [(dynamic)(1.0 / numActions) * (totalReward - output[0])];
        agent.Net.Train(input, excpected, output);
    }
}