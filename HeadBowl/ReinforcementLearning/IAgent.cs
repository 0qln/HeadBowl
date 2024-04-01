namespace HeadBowl.ReinforcementLearning;

public interface IAgent<TPrecision, TEnv, TAction>
    where TPrecision : struct
    where TEnv : IEnviroment<TAction>
{
    public Functions.QFunction<TPrecision, TEnv, TAction> Q { get; }
}