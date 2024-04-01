using HeadBowl.TrainingData.TicTacToe;

namespace HeadBowl.ReinforcementLearning;

public interface IAgent<TPrecision, TEnv, TAction> : IBrain<TPrecision>
    where TPrecision : struct
    where TEnv : IEnviroment<TAction>
{
    /// <summary>
    /// Quality of the State/Action pair.
    /// </summary>
    /// <typeparam name="TEnv"></typeparam>
    /// <typeparam name="TAction"></typeparam>
    /// <returns>The quality of the state s, taken action a.</returns>
    public Functions.QFunction<TPrecision, TEnv, TAction> Q { get; }

    public TPrecision[] GetInputs(TEnv state, TAction action);
}