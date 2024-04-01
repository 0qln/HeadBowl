namespace HeadBowl.ReinforcementLearning;

public interface ITwoAgentEnviroment<TAction> : IEnviroment<TAction>
{
    public int Turn { get; }
    public int Agent1 { get; }
    public int Agent2 { get; }
}