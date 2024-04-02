using System.Numerics;

namespace HeadBowl.ReinforcementLearning;

public static class TemproalDifferenceLearning<TPrecision>
    where TPrecision : struct, ISubtractionOperators<TPrecision, TPrecision, TPrecision>
{
    public static void Update<TState, TAction>(IAgent<TPrecision, TState, TAction> agent, TPrecision totalReward, TState state, TAction action, int numActions)
        where TState : IEnviroment<TAction>
    {
        //TPrecision[] output = [agent.Q(state, action)];
        //TPrecision[] input = agent.GetInputs(state, action);
        //TPrecision[] excpected = 
    }
}