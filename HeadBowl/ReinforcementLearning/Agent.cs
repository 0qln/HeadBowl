
namespace HeadBowl.ReinforcementLearning;

public interface IAgent
{
    public void MakeAction<TAction>(TAction action);
}

public class Agent : IAgent
{
    public void MakeAction<TAction>(TAction action)
    {

    }
}